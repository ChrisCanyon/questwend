#pragma once

// Dynamic per-expert VRAM cache for MoE weight offloading.
//
// MoE models keep only a few experts active per token (e.g. 8 of 256), so the
// hot working set of expert weights is far smaller than the full set. This cache
// keeps pools of expert "slots" resident in VRAM and streams the rest from a
// slower tier (CPU RAM now; SSD later) on demand.
//
// The authoritative expert weights live elsewhere (CPU pinned buffer, owned by
// Model). This class manages VRAM-resident *copies* keyed by (layer,expert) and
// remaps logical expert ids to physical slot ids for ggml_mul_mat_id.
//
// Mixed quantization (e.g. unsloth "UD" quants) gives different layers different
// expert weight *types*, and a single ggml tensor holds exactly one type — so we
// keep one slot pool per (role, type, shape) signature. gate/up/down are pooled
// independently (separate residency), each layer routed to the pool matching its
// own tensor's signature.
//
// ensure() is the single choke point where a cache miss fetches expert bytes.
// To add an SSD tier (Phase C), replace the fetch source inside fetch_slab() with
// a pread from the on-disk weight blob — nothing else changes.

#include <cstdint>
#include <memory>
#include "host_expert_cache.h"
#include "gpu_lru_index.h"
#include "expert_admission.h"
#include "ram_ssd_admission.h"
#include "prefill_cache_policy.h"
#include "expert_ssd_read_phase.h"
#include "activation_policy.h"
#include "expert_telemetry.h"
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct ggml_context;
typedef struct ggml_backend *             ggml_backend_t;
typedef struct ggml_backend_buffer *      ggml_backend_buffer_t;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

namespace questwend {

class Model;
class PersistentExpertIO;
class ExpertWeightAudit;

class ExpertCache {
public:
    enum Role { GATE = 0, UP = 1, DOWN = 2, N_ROLE = 3 };

    // gpu_backend: where the slot pools live.
    // vram_avail_bytes: VRAM budget for the slot pools (resident fraction is
    //   derived so all experts share a uniform residency ratio).
    // ssd: if true, cache misses are streamed from the GGUF file on disk via
    //   pread (experts are not held in RAM); otherwise from the CPU expert tensors.
    // layer_mask (multi-GPU): when given, only the layers it selects get pools
    // here -- the others are computed on another device and cached by that
    // device's ExpertCache. Layer indices stay global either way, so callers
    // index every cache with the same `il`; serves() says which one owns it.
    ExpertCache(ggml_backend_t gpu_backend, Model & model,
                int n_layer, int n_expert, int n_used, size_t vram_avail_bytes,
                bool ssd = false,
                const std::vector<bool> * layer_mask = nullptr);
    ~ExpertCache();

    // Whether this cache holds `layer`'s expert pools.
    bool serves(int layer) const {
        return layer >= 0 && layer < n_layer_ && layer_pool_[GATE][(size_t) layer] >= 0;
    }

    // Make `layer`'s selected experts resident for all three roles, copying
    // misses into VRAM. Fills the three parallel slot-id arrays (length n each).
    void ensure(int layer, const int32_t * expert_ids, int n,
                int32_t * slot_gate, int32_t * slot_up, int32_t * slot_down, bool record_demand = true);

    // Slot-pool tensor for a given layer/role (mul_mat_id weight arg).
    ggml_tensor * tensor(Role role, int layer) const;
    ggml_tensor * gate(int layer) const { return tensor(GATE, layer); }
    ggml_tensor * up(int layer)   const { return tensor(UP,   layer); }
    ggml_tensor * down(int layer) const { return tensor(DOWN, layer); }

    int n_expert() const { return n_expert_; }
    int min_slots() const;   // smallest pool capacity (bounds a safe prefill batch)
    // Fraction of all routed-expert bytes the VRAM pools can hold (0..1).
    double resident_fraction() const { return frac_; }
    // Slot capacity serving `layer` (min over roles): one ensure() call must not
    // reference more distinct experts than this, or it evicts its own slots.
    int capacity(int layer) const;

    // Is (layer,expert) currently resident in the role's pool?
    bool resident(Role role, int layer, int expert) const;

    // Dense "logical expert -> slot" row for (role,layer): n_expert entries,
    // 0 (sentinel) for non-resident experts. Used to fill the in-graph remap
    // table for the optimistic single-graph decode path.
    const int32_t * slot_of_row(Role role, int layer) const;

    // Record a resident access (fast-path hit): updates LRU/frequency/stats
    // without fetching. No-op if not resident.
    void touch(Role role, int layer, int expert);

    // Make all three roles of (layer,expert) resident, fetching any that are
    // absent (counts as misses). Used to repair a speculative single-graph miss.
    void ensure_resident(int layer, int expert);

    struct Stats { uint64_t hits = 0, misses = 0, evictions = 0;
                   double fetch_ms = 0; uint64_t fetch_bytes = 0;
                   bool direct_covering_requested = false;
                   uint64_t direct_covering_batches = 0, direct_covering_fallback_batches = 0;
                   uint64_t direct_covering_read_jobs = 0, direct_covering_payload_bytes = 0;
                   uint64_t direct_covering_requested_bytes = 0;
                   uint64_t pipeline_batches = 0, pipeline_jobs = 0;
                   uint64_t pipeline_ram_bytes = 0, pipeline_file_bytes = 0;
                   double pipeline_owner_wait_ms = 0, pipeline_owner_work_ms = 0; };
    const Stats & stats() const { return stats_; }

    // ---- per-layer instrumentation (QWEN_LAYER_STATS) ----
    // The slot pools are shared by every layer (one pool per role/type/shape),
    // so the split of slots across layers is an emergent property of the global
    // LRU, not a policy. These counters expose that split, plus (under
    // resident-only decode) how often a layer's frozen palette failed to hold
    // what the unmasked router actually wanted.
    struct LayerStat {
        uint64_t hits = 0, misses = 0, evictions = 0;
        uint64_t want = 0, want_miss = 0;
        int      resident = 0;   // experts resident in all three roles
    };
    void layer_stats(std::vector<LayerStat> & out) const;
    void dump_layer_stats(const char * tag) const;
    void reset_layer_stats();
    // resident-decode: the unmasked router wanted (layer,expert); absent =
    // it was not in the resident palette, so the mask routed elsewhere.
    void note_want(int layer, bool absent);

    // ---- per-layer slot quotas (QWEN_LAYER_QUOTA) ----
    // A pool's slots are shared by every layer it serves, and plain global LRU
    // hands them to whichever layer was swept last -- which after a layer-major
    // prefill is the tail of the model. rebalance() instead ranks every observed
    // (layer,expert) in a pool by access count and keeps the top n_slots; the
    // resulting per-layer counts become quotas that eviction then respects. A
    // layer whose routing saturates on a few experts keeps only those, and the
    // slack goes to the diffuse layers that can still convert slots into hits.
    // `fill_budget` bounds how many experts are fetched to reach the new shape.
    void rebalance(size_t fill_budget);
    // Experiment support (QWEN_LAYER_QUOTA_DRIFT): forget the access history,
    // then later ask what quota the *new* history alone would produce. Comparing
    // that to the shape rebalance() actually installed says whether a static
    // quota is enough or the shape has to track the generation.
    void clear_counts();
    void dump_quota_plan(const char * tag) const;
    bool quotas_active() const { return quota_on_; }
    // The layer's quota (min over its role pools), or the pool size when no
    // quota has been computed. Lets a caller ask "does this batch fit the
    // quota?" without having to toggle enforcement to find out.
    int  quota_of(int layer) const;
    // Quotas must be off during batched prefill: one ensure() there references
    // far more distinct experts than any quota, and a quota-bound layer would
    // evict its own slots mid-call (silent corruption).
    void set_quotas(bool on) { quota_on_ = on; }

    // Persist the access-frequency profile (hot experts) for warm restarts, and
    // pre-fill the VRAM slots from a saved profile before generation starts.
    bool   save_profile(const std::string & path) const;
    bool   save_tier_stats(const std::string & path) const;
    bool has_host_cache() const { return host_cache_ != nullptr; }
    struct OwnerStagingCapacity {
        size_t pinned_requested_bytes = 0;
        size_t pageable_vector_capacity_bytes = 0;
    };
    // Explicit owner-thread profile boundary only. No backend calls or buffer
    // reads; excludes worker-private bounce buffers and OS/allocator overhead.
    OwnerStagingCapacity owner_staging_capacity() const {
        OwnerStagingCapacity result;
        result.pinned_requested_bytes = stage_cap_ + coal_stage_cap_ +
            (victim_stage_ ? (64ull << 20) : 0);
        result.pageable_vector_capacity_bytes = stage_.capacity() + coal_buf_.capacity() + abuf_.capacity();
        return result;
    }
    void note_demands(int layer, const int32_t * ids, int count);
    void promote_hot_demand(int layer, const int32_t * ids, int count);
    bool uses_transient_admission() const { return transient_admission_; }
    // Synchronized entry boundary: scratch for this batch selection union, capped at a layer.
    // Kernel batching and source-prefill cache policy are independent.
    void set_phase_workspace(bool batched, int token_count, bool source_prefill = true);
    bool host_resident(int layer, int expert, bool touch = true);
    void request_telemetry_layer(int layer);
    void refresh_telemetry(bool force = false);
    ExpertTelemetry telemetry_snapshot() const;
    // Read-only placement probe used only by the optional debug recorder.
    void debug_placement(std::vector<uint8_t> & gpu, std::vector<uint8_t> & ram) const;
    void activation_lookahead(int current_layer);
    void copy_host_slab(Role role, int layer, int expert, void * destination, size_t bytes);
    std::shared_ptr<const uint8_t[]> borrow_host_slab(Role role, int layer, int expert, size_t bytes);
    size_t load_prefetch(const std::string & path);   // returns # experts prefetched
    void prime_cold(); // opt-in, deterministic empty-cache fill before the first request

private:
    struct Pool {
        ggml_tensor * t = nullptr;          // [ne0, ne1, n_slots], one type
        uint8_t *     host = nullptr;       // host-visible slot memory (unified-memory
                                            // backends only): SSD reads go straight in
        int           role = 0;
        int           scratch_slots = 0;
        int           n_slots = 0;
        std::unordered_map<int, int> key2slot;  // key = layer*n_expert+expert
        std::vector<int>      slot2key;         // -1 = empty
        std::vector<uint64_t> clk;              // LRU timestamps
        std::unique_ptr<GpuLruIndex> lru_index;  // opt-in exact unconstrained scan accelerator
        std::unordered_map<int, uint64_t> count;  // access frequency per key
        std::vector<int32_t>  slot_of;          // key -> slot (0 sentinel if absent)
        std::vector<int>      quota;            // per layer: max slots here (-1 = not served)
        std::vector<int>      occ;              // per layer: slots currently held
    };

    // A reserved-but-not-yet-loaded slot fetch (used for parallel SSD reads).
    struct FetchJob { Pool * pool; int role; int expert; int slot; };

    int  install(Pool & pool, int key, int layer, int expert, Role role); // fetch into a free/LRU slot
    int  slot_for(Pool & pool, int layer, int expert, Role role);
    int  reserve_victim(Pool & pool, int key);   // evict + claim a slot, no fetch
    int  pick_victim(Pool & pool, int layer);    // quota-aware LRU slot choice
    void touch_slot(Pool & pool, int slot);
    void rebuild_gpu_lru_indexes();
    void validate_gpu_lru_indexes() const;
    void compute_quota(std::vector<int> & q) const;  // the greedy, without applying it
    void release_victim(Pool & pool, int slot);
    void drop_slot(Pool & pool, int slot);       // release a slot's key (occ/maps/stats)
    void fetch_slab(Role role, int layer, int expert, ggml_tensor * dst, int slot);
    void fetch_parallel(int layer, std::vector<FetchJob> & jobs);  // SSD: parallel pread + serial H2D
    uint8_t * host_of(const ggml_tensor * t) const;  // pool host ptr for a slot tensor (or null)

    Model &        model_;
    ggml_backend_t backend_ = nullptr;   // GPU backend (for async H2D on its stream)
    double         frac_ = 1.0;          // resident fraction (see resident_fraction())
    bool           async_fetch_ = true;  // overlap RAM-tier H2D with host work
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;

    ExpertSsdReadPhase ssd_read_phase_; // Owner-only; snapshots passed to CPU readers.
    bool  ssd_  = false;
    bool  direct_ = false;                    // Windows: FILE_FLAG_NO_BUFFERING reads
                                              // (bypass the OS page cache; QWEN_SSD_DIRECT)
    std::unordered_map<std::string, void *> hfiles_;  // path -> HANDLE (direct, main thread)
    std::vector<uint8_t> abuf_;               // aligned staging for direct reads (main thread)
    int   prefetch_threads_ = 1;              // parallel SSD read workers
    std::unique_ptr<PersistentExpertIO> io_workers_; // Opt-in persistent singles-path readers
    std::vector<size_t>      foff_[N_ROLE];   // file offset of expert tensor (role,layer)
    std::vector<std::string> fpath_[N_ROLE];  // shard file of expert tensor (role,layer)
    std::unordered_map<std::string, void *> files_;  // path -> FILE* (lazy open)
    std::unordered_map<std::string, int>    fds_;    // path -> fd (zero-copy pread path)

    int n_layer_  = 0;
    int n_expert_ = 0;
    int n_used_   = 0;   // experts per token: the per-layer quota floor
    bool quota_on_ = false;

    std::vector<Pool> pools_;
    // pool index for each (role, layer)
    std::vector<int>  layer_pool_[N_ROLE];

    uint64_t clock_ = 0;
    std::vector<uint8_t> stage_;   // host staging buffer for fetch (pageable fallback)
    // Staging for coalesced (range) SSD reads: pinned when available so the
    // per-slab H2D can be async (the sync pageable copies dominated fetch time).
    std::vector<uint8_t>  coal_buf_;                    // pageable fallback (4KB-align inside)
    ggml_backend_buffer_t coal_stage_buf_ = nullptr;    // pinned staging (or null)
    void *                coal_stage_ptr_ = nullptr;
    size_t                coal_stage_cap_ = 0;
    bool                  coal_pinned_ = false;         // coal_host() returned pinned memory
    bool                  coal_async_pending_ = false;  // async H2D in flight from the staging
    void * coal_host(size_t nbytes);                    // >= nbytes, 4KB-aligned

    // Pinned (page-locked) host staging buffer: the H2D source for a slab upload.
    // Bounded to one slab, so it is always allocatable (unlike pinning all experts)
    // yet gives pinned-DMA bandwidth. Falls back to `stage_` if the backend has no
    // host buffer type (non-CUDA). Allocated lazily on first fetch.
    ggml_backend_buffer_type_t host_buft_ = nullptr;  // pinned host buffer type (or null)
    ggml_backend_buffer_t      stage_buf_ = nullptr;  // pinned staging buffer
    void *                     stage_ptr_ = nullptr;  // base of stage_buf_
    size_t                     stage_cap_ = 0;        // bytes of stage_buf_
    void * stage_host(size_t nbytes);                 // ptr to >= nbytes pinned (or pageable) staging

    std::unique_ptr<ExpertWeightAudit> weight_audit_;
    std::unique_ptr<HostExpertCache> host_cache_;
    std::unique_ptr<ExpertAdmission> admission_;
    RamSsdAdmission ram_ssd_admission_;
    std::unique_ptr<ActivationPolicy> activation_;
    std::atomic<int> telemetry_requested_layer_{-1};
    std::chrono::steady_clock::time_point telemetry_last_{};
    mutable std::mutex telemetry_mutex_;
    ExpertTelemetry telemetry_;
    uint64_t telemetry_total_microseconds_ = 0;
    uint64_t telemetry_samples_ = 0;
    bool infinity_policy_ = false;
    bool activation_scores_ = false, activation_prefetch_ = false;
    double gpu_victim_selection_ms_ = 0, activation_observe_ms_ = 0;
    double prediction_rank_ms_ = 0, prediction_commit_ms_ = 0, prediction_submit_ms_ = 0;
    uint64_t gpu_victim_calls_ = 0, gpu_victim_candidates_ = 0;
    uint64_t unused_retention_decisions_skipped_ = 0; // Owner-only actual covered-role occurrences.
    bool gpu_lru_index_enabled_ = false;
    uint64_t gpu_lru_index_calls_ = 0, gpu_lru_index_candidates_ = 0;
    uint64_t gpu_lru_bitmap_words_ = 0, gpu_lru_index_rebuilds_ = 0;
    double gpu_lru_index_rebuild_ms_ = 0;
    struct PredictedSlab { int role, layer, expert; std::vector<uint8_t> data; };
    std::future<std::vector<PredictedSlab>> prediction_future_;
    std::vector<std::vector<int>> predicted_;
    uint64_t prediction_calls_ = 0, prediction_candidates_ = 0, prediction_hits_ = 0;
    uint64_t prediction_bytes_ = 0, prediction_inserted_bytes_ = 0, prediction_errors_ = 0;
    double prediction_plan_ms_ = 0;
    std::vector<uint8_t> active_demands_;
    bool transient_admission_ = false;
    bool phase_workspace_ = false;
    bool cold_prefill_workspace_ = false;
    bool cold_prefill_pending_ = true;
    PrefillCachePolicy prefill_cache_policy_;
    uint64_t transient_role_fills_ = 0, retained_role_fills_ = 0, proactive_promotions_ = 0;
    bool can_retain(int layer, int expert);
    bool transient_slot(const Pool & pool, int slot) const { return slot >= pool.n_slots; }
    uint64_t ram_key(int role, int layer, int expert) const;
    void demote_to_ram(Pool & pool, int slot, int key);
    void remember_ssd_slab(int role, int layer, int expert, const void * data, size_t bytes, bool retain = false);
    bool readthrough_ram_ = false;
    uint64_t cache_min_free_bytes_ = 2560ull << 20;
    double admission_ms_ = 0;
    uint64_t ram_ssd_rejected_copies_ = 0, ram_ssd_rejected_bytes_ = 0;
    struct PendingVictim { uint64_t key; size_t offset, bytes; };
    std::vector<PendingVictim> pending_victims_;
    ggml_backend_buffer_t victim_stage_ = nullptr;
    size_t victim_used_ = 0;
    uint64_t victim_batches_ = 0;
    bool batch_victims_ = false;
    void flush_victims();
    double demotion_ms_ = 0;
    uint64_t demotion_bytes_ = 0;

    Stats stats_;
    // per-layer counters (indexed by layer); see dump_layer_stats()
    std::vector<uint64_t> l_hits_, l_misses_, l_evict_, l_want_, l_wmiss_;
};

} // namespace questwend
