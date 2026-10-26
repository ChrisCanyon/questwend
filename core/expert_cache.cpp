#include "expert_cache.h"
#include "cache_memory_budget.h"
#include "workspace_partition.h"
#include "gpu_lru_pool_hooks.h"
#include "expert_io_workers.h"
#include "expert_install_check.h"
#include "expert_readiness.h"
#include "expert_weight_audit.h"
#include "model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include "expert_ssd_windows.h"

namespace questwend {

// Protect every expert referenced by the current graph until its slot mapping
// is complete. Nested promotion/ensure calls preserve the outer protection.
struct ActiveExpertGuard {
    std::vector<uint8_t> & flags;
    std::vector<size_t> changed;
    ActiveExpertGuard(std::vector<uint8_t> & protected_ids, int layer, int width,
                     const int32_t * ids, int count) : flags(protected_ids) {
        if (flags.empty()) return;
        for (int index = 0; index < count; ++index) {
            const size_t key = (size_t) layer * width + ids[index];
            if (key >= flags.size()) throw std::out_of_range("Invalid protected expert");
            if (!flags[key]) { flags[key] = 1; changed.push_back(key); }
        }
    }
    ~ActiveExpertGuard() { for (size_t key : changed) flags[key] = 0; }
};

#ifdef __APPLE__
// Is `p` a dereferenceable host pointer (mapped in our address space)?
// Metal shared buffers (unified memory) expose real host memory at
// tensor->data; private buffers hold an unmapped virtual placeholder
// (allocated from 0x400 up, inside PAGEZERO -- never mapped).
static bool ptr_is_mapped(const void * p) {
    if (!p) return false;
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    char vec = 0;
    void * al = (void *) ((uintptr_t) p & ~(uintptr_t) (pg - 1));
    return mincore(al, pg, &vec) == 0;
}
#endif



static ExpertIOClosers expert_io_closers() {
    ExpertIOClosers result;
#ifdef __APPLE__
    result.descriptor = [](int fd) { close(fd); };
#endif
#ifdef _WIN32
    result.direct = [](void * handle) { CloseHandle((HANDLE) handle); };
#endif
    return result;
}

static const char * role_fmt(ExpertCache::Role r) {
    switch (r) {
        case ExpertCache::GATE: return "blk.%d.ffn_gate_exps.weight";
        case ExpertCache::UP:   return "blk.%d.ffn_up_exps.weight";
        default:                return "blk.%d.ffn_down_exps.weight";
    }
}

static ggml_tensor * role_tensor(Model & m, ExpertCache::Role r, int layer) {
    char name[256];
    snprintf(name, sizeof(name), role_fmt(r), layer);
    return m.tensor(name);
}

ExpertCache::ExpertCache(ggml_backend_t gpu_backend, Model & model,
                         int n_layer, int n_expert, int n_used, size_t vram_avail_bytes,
                         bool ssd, const std::vector<bool> * layer_mask)
    : model_(model), ssd_(ssd), n_layer_(n_layer), n_expert_(n_expert) {
    // Layers this cache is responsible for (all of them without a split).
    // A leading dense block (step35, lfm2moe) has no routed experts at all; it
    // is left unserved exactly like another device's layer, so every consumer
    // that already checks serves() skips it.
    auto owns = [&](int il) {
        return (!layer_mask || (*layer_mask)[(size_t) il]) && role_tensor(model, GATE, il) != nullptr;
    };

    backend_ = gpu_backend;
    if (const char * enabled = getenv("QWEN_GPU_LRU_INDEX")) {
        const std::string selected(enabled);
        if (selected != "0" && selected != "1")
            throw std::invalid_argument("QWEN_GPU_LRU_INDEX must be 0 or 1");
        gpu_lru_index_enabled_ = selected == "1";
    }
    if (getenv("QWEN_SYNC_FETCH")) async_fetch_ = false;   // A/B: force synchronous H2D

    if (const char * value = getenv("QWEN_EXPERT_RAM_CACHE_MB")) {
        const size_t mib = std::stoull(value);
        if (mib) {
            if (!ssd_)
                throw std::runtime_error("Bounded RAM victim caching currently requires a local SSD-tier GPU");
            cache_min_free_bytes_ = parse_cache_min_free_mb(getenv("QWEN_CACHE_MIN_FREE_MB"));
            host_cache_ = std::make_unique<HostExpertCache>(mib * 1024ull * 1024ull);
            batch_victims_ = getenv("QWEN_RAM_DEMOTE_BATCH") != nullptr;
            const char * policy = getenv("QWEN_EXPERT_RAM_POLICY");
            readthrough_ram_ = policy && std::string(policy) == "readthrough";
            fprintf(stderr, "expert RAM cache: %zu MiB maximum, %s\n", mib,
                readthrough_ram_ ? "read-through (no GPU downloads)" : "exclusive victim");
        }
    }

    weight_audit_ = ExpertWeightAudit::from_environment(n_layer_);
    if (weight_audit_ && (!ssd_))
        throw std::runtime_error("Expert byte audit requires local SSD-tier execution");

    const char * preserve_decode = getenv("QWEN_PRESERVE_DECODE_CACHE");
    prefill_cache_policy_.set_enabled(preserve_decode && std::string(preserve_decode) != "0");
    transient_admission_ = getenv("QWEN_TINYLFU_ADMISSION") != nullptr;
    phase_workspace_ = transient_admission_ && getenv("QWEN_PHASE_WORKSPACE") != nullptr;
    if (transient_admission_ && (!host_cache_ || readthrough_ram_ || getenv("QWEN_FASTCACHE")))
        throw std::runtime_error("Transient admission requires the exclusive RAM tier and exact segmented execution");
    infinity_policy_ = getenv("QWEN_MOE_INFINITY") != nullptr;
    activation_scores_ = infinity_policy_ && !getenv("QWEN_DISABLE_ACTIVATION_SCORES");
    activation_prefetch_ = infinity_policy_ && !getenv("QWEN_DISABLE_ACTIVATION_PREFETCH");
    // RAM replacement is independent of GPU TinyLFU admission/victim policy.
    if (const char * replacement = getenv("QWEN_RAM_REPLACEMENT")) {
        const std::string selected(replacement);
        if (selected != "lru" && selected != "s3fifo-role-slabs")
            throw std::invalid_argument("QWEN_RAM_REPLACEMENT must be lru or s3fifo-role-slabs");
        if (selected == "s3fifo-role-slabs") {
            if (!host_cache_ || readthrough_ram_)
                throw std::invalid_argument("S3FIFO experiment requires an exclusive RAM victim cache");
            if (activation_scores_)
                throw std::invalid_argument("S3FIFO RAM replacement conflicts with activation scores");
            const uint64_t keys = static_cast<uint64_t>(N_ROLE) * n_layer_ * n_expert_;
            if (!keys || keys >= RamS3FifoMetadata::none)
                throw std::invalid_argument("RAM S3FIFO dense key count is out of range");
            size_t largest_role = 0;
            for (int layer = 0; layer < n_layer_; ++layer) {
                if (!owns(layer)) continue;
                for (int role = 0; role < N_ROLE; ++role)
                    largest_role = std::max(largest_role, role_tensor(model, (Role) role, layer)->nb[2]);
            }
            host_cache_->enable_s3fifo_role_slabs(static_cast<uint32_t>(keys), largest_role);
            fprintf(stderr, "expert RAM replacement: s3fifo-role-slabs (experimental)\n");
        }
    }
    if (infinity_policy_ && !transient_admission_)
        throw std::runtime_error("Activation policy needs the tested exclusive/transient execution path");
    if (infinity_policy_ || getenv("QWEN_ACTIVATION_TRACE")) {
        activation_ = std::make_unique<ActivationPolicy>(n_layer, n_expert);
        predicted_.resize(n_layer);
        if (const char * manifest = getenv("QWEN_ACTIVATION_REFERENCES")) activation_->load_references(manifest);
        if (activation_scores_) host_cache_->set_priority([this](uint64_t key) {
            return activation_->priority(key % ((uint64_t) n_layer_ * n_expert_));
        }, getenv("QWEN_INDEXED_RAM_EVICTION") != nullptr);
    }
    n_used_ = n_used;
    l_hits_.assign(n_layer, 0);  l_misses_.assign(n_layer, 0); l_evict_.assign(n_layer, 0);
    l_want_.assign(n_layer, 0);  l_wmiss_.assign(n_layer, 0);

    // Pinned host staging buffer type (CUDA): makes the slab H2D a pinned-DMA copy.
    if (ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend))
        host_buft_ = ggml_backend_dev_host_buffer_type(dev);

    // SSD tier: precompute each expert tensor's shard file + offset (files are
    // opened lazily on first fetch; sharded models spread experts across files).
    if (ssd_) {
        prefetch_threads_ = 8;
        if (const char * e = getenv("QWEN_PREFETCH_THREADS")) {
            int v = atoi(e);
            if (v >= 1 && v <= 64) prefetch_threads_ = v;
        }
#ifdef _WIN32
        if (getenv("QWEN_SSD_DIRECT")) {
            direct_ = true;
            if (ssd_read_phase_.buffered_source_enabled())
                fprintf(stderr, "expert cache: buffered SOURCE prefill; direct decode/verification\n");
            fprintf(stderr, "expert cache: direct (unbuffered) SSD reads — OS page cache bypassed\n");
        }
#endif
        for (int r = 0; r < N_ROLE; ++r) {
            foff_[r].assign(n_layer, 0);
            fpath_[r].assign(n_layer, std::string());
            for (int il = 0; il < n_layer; ++il) {
                if (!owns(il)) continue;
                char name[256];
                snprintf(name, sizeof(name), role_fmt((Role) r), il);
                foff_[r][il]  = model.tensor_file_offset(name);
                fpath_[r][il] = model.tensor_file(name);
            }
        }
        const char * persistent_io = getenv("QWEN_PERSISTENT_EXPERT_IO");
        if (persistent_io && std::string(persistent_io) != "0") {
            io_workers_ = std::make_unique<PersistentExpertIO>(prefetch_threads_, expert_io_closers());
            fprintf(stderr, "expert I/O: %d persistent workers, worker-local file handles\n", prefetch_threads_);
        }
    }

    // ---- discover (role,type,shape) signatures and route each layer ----
    // Role MUST be part of the signature: gate/up share type & shape but are
    // distinct weights, so they must never land in the same pool.
    struct Sig { int role; int type; int64_t ne0, ne1; size_t slab; int n_layers; };
    std::vector<Sig> sigs;
    for (int r = 0; r < N_ROLE; ++r) {
        layer_pool_[r].assign(n_layer, -1);
        for (int il = 0; il < n_layer; ++il) {
            if (!owns(il)) continue;      // another device's layer: stays -1
            ggml_tensor * t = role_tensor(model, (Role) r, il);
            if (!t) throw std::runtime_error("ExpertCache: missing expert tensor (role " +
                                             std::to_string(r) + ", layer " + std::to_string(il) + ")");
            int idx = -1;
            for (int s = 0; s < (int) sigs.size(); ++s) {
                if (sigs[s].role == r && sigs[s].type == (int) t->type &&
                    sigs[s].ne0 == t->ne[0] && sigs[s].ne1 == t->ne[1]) {
                    idx = s; break;
                }
            }
            if (idx < 0) {
                idx = (int) sigs.size();
                sigs.push_back({ r, (int) t->type, t->ne[0], t->ne[1], t->nb[2], 0 });
            }
            sigs[idx].n_layers++;
            layer_pool_[r][il] = idx;
        }
    }

    // ---- size each pool: uniform residency fraction across all experts ----
    size_t total_bytes = 0;
    for (auto & s : sigs) total_bytes += s.slab * (size_t) s.n_layers * n_expert;
    double frac = total_bytes ? (double) vram_avail_bytes / (double) total_bytes : 1.0;
    if (frac > 1.0) frac = 1.0;
    frac_ = frac;

    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * sigs.size() + 256;
    p.no_alloc = true;
    ctx_ = ggml_init(p);

    pools_.resize(sigs.size());
    for (int s = 0; s < (int) sigs.size(); ++s) {
        int cap = sigs[s].n_layers * n_expert;
        int n_slots = (int) (frac * cap);
        // Correctness floor: one ensure() needs n_used distinct slots in a pool.
        // Fewer would thrash within a single layer and corrupt the matmul.
        if (n_slots < n_used) n_slots = n_used;
        if (n_slots > cap)    n_slots = cap;

        ggml_tensor * t = ggml_new_tensor_3d(ctx_, (ggml_type) sigs[s].type,
                                             sigs[s].ne0, sigs[s].ne1, n_slots);
        char nm[64]; snprintf(nm, sizeof(nm), "expcache.pool%d", s);
        ggml_set_name(t, nm);

        pools_[s].t = t;
        pools_[s].role = sigs[s].role;
        // Physical pool size stays within the existing GPU budget. Reserve a
        // full layer's possible experts as transient workspace so batched
        // prefill can bypass admission without serializing into single tokens.
        pools_[s].scratch_slots = transient_admission_ ? n_expert : 0;
        if (n_slots <= pools_[s].scratch_slots + n_used)
            throw std::runtime_error("GPU pool cannot fit transient workspace plus retained experts");
        pools_[s].n_slots = n_slots - pools_[s].scratch_slots;
        pools_[s].slot2key.assign(pools_[s].n_slots, -1);
        pools_[s].clk.assign(pools_[s].n_slots, 0);
        pools_[s].slot_of.assign((size_t) n_layer * n_expert, 0);
        pools_[s].quota.assign(n_layer, -1);
        pools_[s].occ.assign(n_layer, 0);
        if (gpu_lru_index_enabled_)
            pools_[s].lru_index = std::make_unique<GpuLruIndex>(static_cast<size_t>(t->ne[2]));
    }
    rebuild_gpu_lru_indexes();

    if (transient_admission_) {
        size_t role_slots = 0;
        for (const auto & pool : pools_) role_slots += pool.n_slots;
        admission_ = std::make_unique<ExpertAdmission>((size_t) n_layer * n_expert, role_slots / 3);
        active_demands_.assign((size_t) n_layer * n_expert, 0);
        fprintf(stderr, "expert admission: TinyLFU dense demand counts, transient GPU workspace, whole-expert eviction\n");
    }
    // Capture this opt-in policy once. An enabled policy requires the existing
    // logical demand counters; a missing counter must not reject every expert.
    ram_ssd_admission_ = RamSsdAdmission::from_environment(admission_ != nullptr);
    if (ram_ssd_admission_.enabled())
        fprintf(stderr, "RAM SSD admission: minimum %u logical hits\n", ram_ssd_admission_.minimum_hits());
    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, gpu_backend);
    if (!buf_) {
        size_t want = 0;
        for (int s = 0; s < (int) sigs.size(); ++s) want += sigs[s].slab * (size_t) pools_[s].n_slots;
        throw_alloc_failure(ggml_backend_get_default_buffer_type(gpu_backend), want,
                            "the expert slot pools");
    }
    ggml_backend_buffer_set_usage(buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

#ifdef __APPLE__
    // Unified memory (Metal shared buffers): the slot pools are host-writable,
    // so SSD misses can be pread() straight into the slot -- no staging buffer,
    // no memcpy, no stdio buffering.
    if (ssd_ && !getenv("QWEN_NO_DIRECT_FETCH")) {
        int n_direct = 0;
        for (auto & p : pools_)
            if (ptr_is_mapped(p.t->data)) { p.host = (uint8_t *) p.t->data; ++n_direct; }
        if (n_direct)
            fprintf(stderr, "expert cache: zero-copy SSD reads (unified memory, %d/%zu pools)\n",
                    n_direct, pools_.size());
    }
#endif

    size_t resident = 0;
    for (int s = 0; s < (int) sigs.size(); ++s)
        resident += sigs[s].slab * (size_t) pools_[s].n_slots;
    fprintf(stderr,
            "expert cache (dynamic): %zu pools, %.0f%% resident = %.1f MB VRAM "
            "(of %.1f MB experts, %d layers x %d experts)\n",
            sigs.size(), frac * 100.0, resident / 1024.0 / 1024.0,
            total_bytes / 1024.0 / 1024.0, n_layer, n_expert);
}

ExpertCache::~ExpertCache() {
    // Speculation owns private buffers and file handles. It never touches CUDA
    // or managed residency from its thread; finish before destroying metadata.
    if (prediction_future_.valid()) prediction_future_.wait();
    io_workers_.reset(); // Join CPU I/O workers before their captured metadata/staging can be destroyed.
    // Normal fetches drain all victims before overwriting a GPU slot. A failed
    // fetch can leave queued downloads; drain before releasing their buffer.
    if (!pending_victims_.empty()) ggml_backend_synchronize(backend_);
    if (victim_stage_) ggml_backend_buffer_free(victim_stage_);
    for (auto & kv : files_) if (kv.second) fclose((FILE *) kv.second);
#ifdef __APPLE__
    for (auto & kv : fds_) if (kv.second >= 0) close(kv.second);
#endif
#ifdef _WIN32
    for (auto & kv : hfiles_) if (kv.second) CloseHandle((HANDLE) kv.second);
#endif
    if (coal_stage_buf_) ggml_backend_buffer_free(coal_stage_buf_);
    if (stage_buf_) ggml_backend_buffer_free(stage_buf_);
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

uint8_t * ExpertCache::host_of(const ggml_tensor * t) const {
    for (const auto & p : pools_) if (p.t == t) return p.host;
    return nullptr;
}

// Return a host pointer to at least `nbytes` of staging memory. Prefers a pinned
// (page-locked) buffer for fast H2D; falls back to a pageable vector if the
// backend has no host buffer type or the pinned allocation fails.
void * ExpertCache::stage_host(size_t nbytes) {
    if (host_buft_) {
        if (stage_cap_ < nbytes) {
            if (stage_buf_) ggml_backend_buffer_free(stage_buf_);
            stage_buf_ = ggml_backend_buft_alloc_buffer(host_buft_, nbytes);
            if (stage_buf_) {
                stage_ptr_ = ggml_backend_buffer_get_base(stage_buf_);
                stage_cap_ = nbytes;
            } else {
                host_buft_ = nullptr;   // pinned alloc failed: stop trying
                stage_ptr_ = nullptr;
                stage_cap_ = 0;
            }
        }
        if (stage_ptr_) return stage_ptr_;
    }
    if (stage_.size() < nbytes) stage_.resize(nbytes);
    return stage_.data();
}

// 4KB-aligned staging for coalesced reads; pinned (page-locked) when the
// backend offers a host buffer type, so slab uploads can be true async DMA.
void * ExpertCache::coal_host(size_t nbytes) {
    if (host_buft_ && coal_stage_cap_ < nbytes) {
        if (coal_stage_buf_) ggml_backend_buffer_free(coal_stage_buf_);
        coal_stage_buf_ = ggml_backend_buft_alloc_buffer(host_buft_, nbytes);
        if (coal_stage_buf_) {
            coal_stage_ptr_ = ggml_backend_buffer_get_base(coal_stage_buf_);
            coal_stage_cap_ = nbytes;
        } else {
            coal_stage_ptr_ = nullptr;
            coal_stage_cap_ = 0;
        }
    }
    if (coal_stage_ptr_ && coal_stage_cap_ >= nbytes) { coal_pinned_ = true; return coal_stage_ptr_; }
    coal_pinned_ = false;
    if (coal_buf_.size() < nbytes + 4096) coal_buf_.resize(nbytes + 4096);
    return (void *) (((uintptr_t) coal_buf_.data() + 4095) & ~(uintptr_t) 4095);
}

ggml_tensor * ExpertCache::tensor(Role role, int layer) const {
    if (!serves(layer))
        throw std::runtime_error("ExpertCache::tensor on layer " + std::to_string(layer) +
                                 " which this device does not serve");
    return pools_[layer_pool_[role][layer]].t;
}

void ExpertCache::set_phase_workspace(bool batched, int token_count, bool source_prefill) {
    // Owner boundary: all prior foreground CPU readers have joined.
    // Record genuine source phase even if the physical partition stays unchanged.
    ssd_read_phase_.enter_phase(source_prefill);
    // Record both phases even when physical workspace resizing is disabled.
    prefill_cache_policy_.enter_phase(source_prefill);
    if (!phase_workspace_) return;
    const int scratch = bounded_workspace_scratch(n_expert_, n_used_, batched ? token_count : 1);
    const auto plan = plan_workspace_partition(pools_, scratch, n_used_);
    if (!plan.changed) return;
    require_workspace_boundary(active_demands_);
    // Allocate metadata before demoting any weights. Physical tensors and their
    // dimensions are unchanged, so persistent graphs still reference the same buffers.
    for (size_t index = 0; index < pools_.size(); ++index) {
        auto & pool = pools_[index];
        if (pool.t->ne[2] != pool.n_slots + pool.scratch_slots)
            throw std::logic_error("Physical expert pool size changed");
        pool.slot2key.reserve(plan.retained[index]);
        pool.clk.reserve(plan.retained[index]);
    }
    flush_victims();
    ggml_backend_synchronize(backend_);
    coal_async_pending_ = false;
    retire_workspace_tail(pools_, plan, [this](size_t index, int slot) {
        release_victim(pools_[index], slot);
    });
    // Every demotion must finish before the old retained tail becomes scratch.
    flush_victims();
    ggml_backend_synchronize(backend_);
    coal_async_pending_ = false;
    commit_workspace_partition(pools_, plan);
    // Retired tails were removed using the old prefix. Rebuild only after every
    // retained vector has its new size; the physical pool/index allocation stays.
    rebuild_gpu_lru_indexes();
    size_t role_slots = 0, scratch_bytes = 0;
    for (auto & pool : pools_) {
        role_slots += pool.n_slots;
        scratch_bytes += (size_t) pool.scratch_slots * pool.t->nb[2];
        // These quotas were derived from the previous retained capacity. Keep
        // demand counts but use unconstrained LRU until a later rebalance.
        std::fill(pool.quota.begin(), pool.quota.end(), -1);
    }
    quota_on_ = false;
    // Do not age preserved decode history against the temporary, smaller
    // prefill retained capacity. Decode restores its own horizon.
    if (prefill_cache_policy_.record_decode_frequency())
        admission_->set_capacity(role_slots / N_ROLE);
    fprintf(stderr, "expert workspace: %s, %d scratch slots per pool, %.1f MiB transient\n",
        batched ? "batch" : "decode", scratch, scratch_bytes / 1048576.0);
}

int ExpertCache::min_slots() const {
    int m = pools_.empty() ? 0 : pools_[0].n_slots;
    for (const auto & p : pools_) m = std::min(m, p.n_slots);
    return m;
}

int ExpertCache::capacity(int layer) const {
    if (!serves(layer)) return 0;
    int m = INT32_MAX;
    for (int r = 0; r < N_ROLE; ++r) {
        const Pool & p = pools_[layer_pool_[r][layer]];
        // With quotas on, the layer can only hold its quota before it starts
        // recycling its own slots -- that, not the pool size, is what an
        // ensure() call must stay under.
        const int c = quota_on_ && p.quota[layer] >= 0
                          ? std::max(p.quota[layer], n_used_) : p.n_slots;
        m = std::min(m, c);
    }
    return m;
}

int ExpertCache::quota_of(int layer) const {
    if (!serves(layer)) return 0;
    int m = INT32_MAX;
    for (int r = 0; r < N_ROLE; ++r) {
        const Pool & p = pools_[layer_pool_[r][layer]];
        m = std::min(m, p.quota[layer] >= 0 ? p.quota[layer] : p.n_slots);
    }
    return m;
}

// Rank a pool's observed (layer,expert) pairs by access count, keep the top
// n_slots, and turn the per-layer counts of that ideal set into quotas. This is
// the miss-minimising static split: a slot is worth the number of accesses it
// would serve, so a layer whose router concentrates on a handful of experts
// keeps only those, while a diffuse layer -- whose experts are individually
// lukewarm but numerous -- takes the slack.
void ExpertCache::compute_quota(std::vector<int> & quota) const {
    // Access counts are role-independent (ensure() touches all three roles for
    // the same selection), but the roles can live in pools of different sizes.
    // A resident expert is only usable when all three roles are present, so the
    // quota must be ONE number per layer that every one of its pools can honour
    // -- computing it per pool hands the same layer three different quotas and
    // the intersection collapses.
    quota.assign(n_layer_, 0);
    std::unordered_map<int, uint64_t> cnt;
    for (const auto & pool : pools_)
        for (const auto & kv : pool.count) cnt[kv.first] += kv.second;
    if (cnt.empty()) return;

    std::vector<std::pair<uint64_t, int>> ents;   // (count, key)
    ents.reserve(cnt.size());
    for (const auto & kv : cnt) ents.emplace_back(kv.second, kv.first);
    std::sort(ents.begin(), ents.end(), std::greater<std::pair<uint64_t, int>>());

    // A layer consumes one slot in each of its three pools per expert it holds.
    std::vector<int> rem(pools_.size());
    for (size_t i = 0; i < pools_.size(); ++i) rem[i] = pools_[i].n_slots;
    auto take = [&](int il) {
        if (!serves(il)) return false;      // layer belongs to another device
        int pi[N_ROLE];
        for (int r = 0; r < N_ROLE; ++r) {
            pi[r] = layer_pool_[r][il];
            if (rem[pi[r]] <= 0) return false;
        }
        for (int r = 0; r < N_ROLE; ++r) rem[pi[r]]--;
        return true;
    };

    // Correctness floor first: every layer must be able to hold one token's
    // selection, or a single ensure() evicts its own slots.
    for (int il = 0; il < n_layer_; ++il)
        while (quota[il] < n_used_ && take(il)) quota[il]++;

    // Then the miss-minimising greedy: a slot is worth the accesses it serves,
    // so the layers whose routing spreads over many lukewarm experts keep
    // winning slots after the concentrated layers have run out of hot ones.
    std::vector<int> seen(n_layer_, 0);
    for (const auto & e : ents) {
        const int il = e.second / n_expert_;
        if (seen[il]++ < n_used_) continue;        // already paid for by the floor
        if (quota[il] >= n_expert_) continue;
        if (!take(il)) continue;
        quota[il]++;
    }
}

void ExpertCache::clear_counts() {
    for (auto & pool : pools_) pool.count.clear();
}

void ExpertCache::dump_quota_plan(const char * tag) const {
    std::vector<int> q;
    compute_quota(q);
    fprintf(stderr, "quota-plan[%s]:", tag);
    for (int il = 0; il < n_layer_; ++il) if (serves(il)) fprintf(stderr, " %d", q[il]);
    fprintf(stderr, "\n");
}

void ExpertCache::rebalance(size_t fill_budget) {
    std::vector<int> quota;
    compute_quota(quota);
    if (quota.empty()) return;

    // ents is needed again below to drive the reshape in count order.
    std::unordered_map<int, uint64_t> cnt;
    for (const auto & pool : pools_)
        for (const auto & kv : pool.count) cnt[kv.first] += kv.second;
    std::vector<std::pair<uint64_t, int>> ents;
    ents.reserve(cnt.size());
    for (const auto & kv : cnt) ents.emplace_back(kv.second, kv.first);
    std::sort(ents.begin(), ents.end(), std::greater<std::pair<uint64_t, int>>());
    std::vector<int> seen(n_layer_, 0);

    size_t moved = 0;
    for (int il = 0; il < n_layer_; ++il) {
        if (!serves(il)) continue;
        for (int r = 0; r < N_ROLE; ++r) {
            Pool & pool = pools_[layer_pool_[r][il]];
            if (pool.quota[il] != quota[il]) { pool.quota[il] = quota[il]; ++moved; }
        }
    }

    // Reshape residency towards the ideal set. The quota alone only steers
    // future evictions, and after a layer-major prefill the under-quota layers
    // hold nothing at all -- they would need thousands of decode tokens to
    // claim what is already theirs. All three roles are installed together so
    // the expert is actually usable.
    quota_on_ = true;
    size_t filled = 0;
    std::fill(seen.begin(), seen.end(), 0);
    for (const auto & e : ents) {
        if (filled >= fill_budget) break;
        const int key = e.second, il = key / n_expert_, ex = key % n_expert_;
        if (!serves(il)) continue;
        if (seen[il] >= quota[il]) continue;
        seen[il]++;
        bool all = true;
        for (int r = 0; r < N_ROLE; ++r)
            all = all && pools_[layer_pool_[r][il]].key2slot.count(key);
        if (all) continue;
        for (int r = 0; r < N_ROLE; ++r) {
            Pool & pool = pools_[layer_pool_[r][il]];
            if (!pool.key2slot.count(key)) install(pool, key, il, ex, (Role) r);
        }
        ++filled;
    }

    ggml_backend_synchronize(backend_);
    int qmin = n_expert_, qmax = 0;
    for (int il = 0; il < n_layer_; ++il) {
        if (!serves(il)) continue;
        qmin = std::min(qmin, quota[il]); qmax = std::max(qmax, quota[il]);
    }
    fprintf(stderr, "expert cache: per-layer quotas on (%zu role-quotas set, range %d..%d experts), "
                    "%zu experts prefetched into the new shape\n", moved, qmin, qmax, filled);
    if (getenv("QWEN_LAYER_QUOTA_DUMP")) {
        fprintf(stderr, "quota:");
        for (int il = 0; il < n_layer_; ++il) fprintf(stderr, " %d", quota[il]);
        fprintf(stderr, "\n");
    }
}

// Copy one expert's slab from the slower tier into a VRAM slot.
// This is the tiering seam: RAM (tensor_get) vs SSD (pread from the GGUF file).
void ExpertCache::fetch_slab(Role role, int layer, int expert, ggml_tensor * dst, int slot) {
    const auto io_phase = ssd_read_phase_.begin_fetch(direct_);
    const bool direct_read = io_phase.direct();
    flush_victims();
    const auto t0 = std::chrono::steady_clock::now();
    ggml_tensor * src = role_tensor(model_, role, layer);
    const size_t nb2 = src->nb[2];
    if (dst->nb[2] != nb2)
        throw std::runtime_error("ExpertCache: slab size mismatch (pool vs source)");

    const void * hsrc;
    if (ssd_) {
        const size_t off = foff_[role][layer] + (size_t) expert * nb2;

#ifdef __APPLE__
        // unified memory: pread straight into the slot (no staging, no memcpy)
        if (uint8_t * hp = host_of(dst)) {
            const std::string & path = fpath_[role][layer];
            auto it = fds_.find(path);
            int fd;
            if (it == fds_.end()) {
                fd = open(path.c_str(), O_RDONLY);
                fds_[path] = fd;
            } else fd = it->second;
            if (fd < 0) throw std::runtime_error("ExpertCache: failed to open shard: " + path);
            if (pread(fd, hp + (size_t) slot * nb2, nb2, (off_t) off) != (ssize_t) nb2)
                throw std::runtime_error("ExpertCache: SSD pread failed");
            stats_.fetch_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            stats_.fetch_bytes += nb2;
            return;
        }
#endif
        // SSD tier: read the slab into a pinned staging buffer, then H2D from it.
        void * stage = stage_host(nb2);
        if (host_cache_ && (readthrough_ram_ ? host_cache_->read(ram_key(role, layer, expert), stage, nb2) : host_cache_->take(ram_key(role, layer, expert), stage, nb2))) {
            ggml_backend_tensor_set(dst, stage, (size_t) slot * nb2, nb2);
            stats_.fetch_bytes += nb2;
            stats_.fetch_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            return;
        }
#ifdef _WIN32
        if (direct_read) {
            void *& hv = hfiles_[fpath_[role][layer]];
            if (!hv) hv = win_direct_open(fpath_[role][layer]);
            if (!hv || !win_direct_read(hv, stage, nb2, off, abuf_))
                throw std::runtime_error("ExpertCache: direct SSD read failed");
            hsrc = stage;
        } else {
#endif
        void *& fp = files_[fpath_[role][layer]];
        if (!fp) {
            fp = (void *) fopen(fpath_[role][layer].c_str(), "rb");
            if (!fp) throw std::runtime_error("ExpertCache: failed to open shard: " + fpath_[role][layer]);
        }
        FILE * f = (FILE *) fp;
#ifdef _WIN32
        if (_fseeki64(f, (long long) off, SEEK_SET) != 0 ||
#else
        if (fseeko(f, (off_t) off, SEEK_SET) != 0 ||
#endif
            fread(stage, 1, nb2, f) != nb2)
            throw std::runtime_error("ExpertCache: SSD pread failed");
        hsrc = stage;
#ifdef _WIN32
        }
#endif
        remember_ssd_slab(role, layer, expert, stage, nb2);
    } else {
        // RAM tier: the expert weights live in a (pinned) CPU buffer -> H2D straight
        // from the source, no intermediate staging copy.
        hsrc = (const char *) src->data + (size_t) expert * nb2;
        if (async_fetch_) {
            // Async H2D on the backend's stream: the host does not block per copy,
            // and the following seg-B compute (same stream) is naturally ordered
            // after it. Source is pinned (persistent), so this is a true async DMA.
            ggml_backend_tensor_set_async(backend_, dst, hsrc, (size_t) slot * nb2, nb2);
            stats_.fetch_bytes += nb2;
            return;
        }
    }
    ggml_backend_tensor_set(dst, hsrc, (size_t) slot * nb2, nb2);
    stats_.fetch_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    stats_.fetch_bytes += nb2;
}

uint64_t ExpertCache::ram_key(int role, int layer, int expert) const {
    return ((uint64_t) role * n_layer_ + layer) * n_expert_ + expert;
}

bool ExpertCache::host_resident(int layer, int expert, bool touch) {
    if (!host_cache_) return false;
    for (int role = 0; role < N_ROLE; ++role)
        if (!(touch ? host_cache_->contains_and_touch(ram_key(role, layer, expert)) :
                      host_cache_->contains(ram_key(role, layer, expert)))) return false;
    return true;
}

void ExpertCache::copy_host_slab(Role role, int layer, int expert, void * destination, size_t bytes) {
    if (!host_cache_ || !host_cache_->copy(ram_key(role, layer, expert), destination, bytes))
        throw std::runtime_error("CPU expert lost a required RAM slab before execution");
}

std::shared_ptr<const uint8_t[]> ExpertCache::borrow_host_slab(Role role, int layer, int expert, size_t bytes) {
    auto view = host_cache_ ? host_cache_->borrow(ram_key(role, layer, expert), bytes) : nullptr;
    if (!view) throw std::runtime_error("CPU expert lost a required RAM slab before borrowing");
    return view;
}

void ExpertCache::remember_ssd_slab(int role, int layer, int expert, const void * data, size_t bytes, bool retain) {
    if (!host_cache_ || (!readthrough_ram_ && !retain)) return;
    const uint64_t key = ram_key(role, layer, expert);
    if (host_cache_->contains_and_touch(key)) return;
    // Existing RAM reads/takes retain their normal semantics. New SSD
    // prefill copies execute in GPU slots but need not evict decode-hot RAM.
    if (!prefill_cache_policy_.admit_new_ssd_copy()) return;
    // All three roles consult the same logical expert count. This gates only
    // new optional SSD-to-RAM copies; GPU computation, demotions and leases stay
    // unchanged. Existing roles were touched above, before this predicate.
    if (ram_ssd_admission_.enabled() && !ram_ssd_admission_.accepts(
            admission_->frequency((size_t) layer * n_expert_ + expert))) {
        ++ram_ssd_rejected_copies_;
        ram_ssd_rejected_bytes_ += bytes;
        return;
    }
#ifdef _WIN32
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (cache_min_free_bytes_ > 0 && GlobalMemoryStatusEx(&memory) &&
        cache_headroom_exhausted(memory.ullAvailPhys, bytes, cache_min_free_bytes_)) {
        const auto saved = host_cache_->stats();
        host_cache_->trim(saved.resident_bytes - std::min<uint64_t>(saved.resident_bytes, 512ull << 20));
        return;
    }
#endif
    const auto started = std::chrono::steady_clock::now();
    host_cache_->put(key, bytes, [&](void * destination) { std::memcpy(destination, data, bytes); });
    admission_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void ExpertCache::flush_victims() {
    if (pending_victims_.empty()) return;
    const auto started = std::chrono::steady_clock::now();
    ggml_backend_synchronize(backend_);
    coal_async_pending_ = false;
    const auto * source = (const uint8_t *) ggml_backend_buffer_get_base(victim_stage_);
    for (const auto & victim : pending_victims_) {
        const bool admitted = host_cache_->put(victim.key, victim.bytes,
            [&](void * destination) { std::memcpy(destination, source + victim.offset, victim.bytes); });
        if (admitted) demotion_bytes_ += victim.bytes;
    }
    pending_victims_.clear();
    victim_used_ = 0;
    ++victim_batches_;
    demotion_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void ExpertCache::demote_to_ram(Pool & pool, int slot, int key) {
    if (!host_cache_ || readthrough_ram_) return;
    const size_t bytes = pool.t->nb[2];
#ifdef _WIN32
    // Honor the configurable system-headroom reserve (default 2560 MiB).
    // Explicit zero leaves only the independent cache-capacity bound.
    // Otherwise trim a bounded batch; SSD remains the immutable backing.
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (cache_min_free_bytes_ > 0 && GlobalMemoryStatusEx(&memory) &&
        cache_headroom_exhausted(memory.ullAvailPhys, bytes, cache_min_free_bytes_)) {
        const auto saved = host_cache_->stats();
        const size_t trim = std::min<uint64_t>(saved.resident_bytes, 512ull << 20);
        host_cache_->trim((size_t) saved.resident_bytes - trim);
        return;
    }
#endif
    constexpr size_t VICTIM_BUFFER_BYTES = 64ull << 20;
    if (batch_victims_ && host_buft_ && bytes <= VICTIM_BUFFER_BYTES) {
        if (!victim_stage_) victim_stage_ = ggml_backend_buft_alloc_buffer(host_buft_, VICTIM_BUFFER_BYTES);
        if (victim_stage_) {
            if (victim_used_ + bytes > VICTIM_BUFFER_BYTES) flush_victims();
            auto * destination = (uint8_t *) ggml_backend_buffer_get_base(victim_stage_) + victim_used_;
            ggml_backend_tensor_get_async(backend_, pool.t, destination, (size_t) slot * bytes, bytes);
            pending_victims_.push_back({ram_key(pool.role, key / n_expert_, key % n_expert_), victim_used_, bytes});
            victim_used_ += bytes;
            return;
        }
    }
    const auto started = std::chrono::steady_clock::now();
    const bool admitted = host_cache_->put(ram_key(pool.role, key / n_expert_, key % n_expert_), bytes,
        [&](void * destination) {
            if (coal_async_pending_) {
                ggml_backend_synchronize(backend_);
                coal_async_pending_ = false;
            }
            ggml_backend_tensor_get(pool.t, destination, (size_t) slot * bytes, bytes);
        });
    if (admitted) demotion_bytes_ += bytes;
    demotion_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

bool ExpertCache::save_tier_stats(const std::string & path) const {
    validate_gpu_lru_indexes(); // Explicit owner profile boundary; no hot-path full scan.
    if (activation_) {
        activation_->save_trace(path + ".activation.tsv");
        std::ofstream metrics(path + ".activation.json");
        metrics << "{\"policy_enabled\":" << (infinity_policy_ ? "true" : "false")
                << ",\"prediction_calls\":" << prediction_calls_
                << ",\"predicted_experts\":" << prediction_candidates_
                << ",\"next_layer_hits\":" << prediction_hits_
                << ",\"bytes_read\":" << prediction_bytes_
                << ",\"bytes_inserted\":" << prediction_inserted_bytes_
                << ",\"prefetch_errors\":" << prediction_errors_
                << ",\"planning_ms\":" << prediction_plan_ms_
                << ",\"ranking_ms\":" << prediction_rank_ms_
                << ",\"commit_ms\":" << prediction_commit_ms_
                << ",\"submit_ms\":" << prediction_submit_ms_
                << ",\"observe_ms\":" << activation_observe_ms_
                << ",\"gpu_victim_ms\":" << gpu_victim_selection_ms_
                << ",\"gpu_victim_calls\":" << gpu_victim_calls_
                << ",\"gpu_victim_candidates\":" << gpu_victim_candidates_
                << ",\"ram_victim_ms\":" << (host_cache_ ? host_cache_->stats().eviction_selection_ms : 0)
                << ",\"ram_victim_candidates\":" << (host_cache_ ? host_cache_->stats().eviction_candidates : 0)
                << ",\"ram_priority_refresh_ms\":" << (host_cache_ ? host_cache_->stats().priority_refresh_ms : 0)
                << ",\"ram_priority_refresh_entries\":" << (host_cache_ ? host_cache_->stats().priority_refresh_entries : 0)
                << ",\"scores_enabled\":" << (activation_scores_ ? "true" : "false")
                << ",\"prefetch_enabled\":" << (activation_prefetch_ ? "true" : "false") << "}\n";
    }
    uint64_t partial = 0, overlapping_slabs = 0, scratch_bytes = 0;
    uint64_t complete_ram_experts = 0, partial_ram_slabs = 0;
    for (const auto & pool : pools_) scratch_bytes += (uint64_t) pool.scratch_slots * pool.t->nb[2];
    if (admission_) {
        for (int layer = 0; layer < n_layer_; ++layer) {
            if (!serves(layer)) continue;
            for (int expert = 0; expert < n_expert_; ++expert) {
                int roles = 0, ram_roles = 0;
                for (int role = 0; role < N_ROLE; ++role) {
                    const bool on_gpu = resident((Role) role, layer, expert);
                    roles += on_gpu;
                    const bool in_ram = host_cache_->contains(ram_key(role, layer, expert));
                    ram_roles += in_ram;
                    if (on_gpu && in_ram) ++overlapping_slabs;
                }
                if (roles != 0 && roles != N_ROLE) ++partial;
                if (ram_roles == N_ROLE) ++complete_ram_experts;
                else partial_ram_slabs += ram_roles;
            }
        }
        if (partial || overlapping_slabs)
            throw std::runtime_error("Exclusive whole-expert cache ownership failed");
    }
    FILE * file = fopen(path.c_str(), "wb");
    if (!file) return false;
    const auto saved = host_cache_ ? host_cache_->stats() : HostExpertCache::Stats{};
    fprintf(file, "{\"enabled\":%s,\"capacity_bytes\":%llu,\"resident_bytes\":%llu,"
        "\"peak_bytes\":%llu,\"hits\":%llu,\"misses\":%llu,\"hit_bytes\":%llu,"
        "\"admissions\":%llu,\"admitted_bytes\":%llu,\"evictions\":%llu,"
        "\"evicted_bytes\":%llu,\"demotion_bytes\":%llu,\"demotion_ms\":%.3f,"
        "\"cpu_reads\":%llu,\"cpu_read_bytes\":%llu,\"admission_ms\":%.3f,\"fetch_ms\":%.3f,\"fetch_bytes\":%llu,\"victim_batches\":%llu,"
        "\"pooled_bytes\":%llu,\"allocations\":%llu,\"reused_allocations\":%llu,\"allocation_ms\":%.3f,\"fill_ms\":%.3f,\"cpu_copy_ms\":%.3f,\"transient_role_fills\":%llu,\"retained_role_fills\":%llu,\"proactive_promotions\":%llu,\"partial_gpu_experts\":%llu,\"overlapping_managed_slabs\":%llu,\"transient_workspace_bytes\":%llu,\"complete_ram_experts\":%llu,\"partial_ram_slabs\":%llu,\"retired_leased_bytes\":%llu,\"ram_ssd_min_hits\":%u,\"ram_ssd_rejected_copies\":%llu,\"ram_ssd_rejected_bytes\":%llu,\"pipeline_batches\":%llu,\"pipeline_jobs\":%llu,\"pipeline_ram_bytes\":%llu,\"pipeline_file_bytes\":%llu,\"pipeline_owner_wait_ms\":%.3f,\"pipeline_owner_work_ms\":%.3f,\"policy\":\"%s\"",
        host_cache_ ? "true" : "false", (unsigned long long) saved.capacity_bytes,
        (unsigned long long) saved.resident_bytes, (unsigned long long) saved.peak_bytes,
        (unsigned long long) saved.hits, (unsigned long long) saved.misses,
        (unsigned long long) saved.hit_bytes, (unsigned long long) saved.admissions,
        (unsigned long long) saved.admitted_bytes, (unsigned long long) saved.evictions,
        (unsigned long long) saved.evicted_bytes, (unsigned long long) demotion_bytes_, demotion_ms_,
        (unsigned long long) saved.cpu_reads, (unsigned long long) saved.cpu_read_bytes, admission_ms_,
        stats_.fetch_ms, (unsigned long long) stats_.fetch_bytes, (unsigned long long) victim_batches_,
        (unsigned long long) saved.pooled_bytes, (unsigned long long) saved.allocations,
        (unsigned long long) saved.reused_allocations, saved.allocation_ms, saved.fill_ms, saved.cpu_copy_ms,
        (unsigned long long) transient_role_fills_, (unsigned long long) retained_role_fills_,
        (unsigned long long) proactive_promotions_, (unsigned long long) partial,
        (unsigned long long) overlapping_slabs, (unsigned long long) scratch_bytes,
        (unsigned long long) complete_ram_experts, (unsigned long long) partial_ram_slabs,
        (unsigned long long) saved.retired_leased_bytes,
        ram_ssd_admission_.minimum_hits(), (unsigned long long) ram_ssd_rejected_copies_,
        (unsigned long long) ram_ssd_rejected_bytes_,
        (unsigned long long) stats_.pipeline_batches, (unsigned long long) stats_.pipeline_jobs,
        (unsigned long long) stats_.pipeline_ram_bytes, (unsigned long long) stats_.pipeline_file_bytes,
        stats_.pipeline_owner_wait_ms, stats_.pipeline_owner_work_ms,
        admission_ ? "tiny_lfu_admission_exclusive_experts" : readthrough_ram_ ? "readthrough_lru" : "exclusive_victim_lru");
    fprintf(file, ",\"direct_covering_requested\":%s,\"direct_covering_batches\":%llu,"
        "\"direct_covering_fallback_batches\":%llu,\"direct_covering_read_jobs\":%llu,"
        "\"direct_covering_payload_bytes\":%llu,\"direct_covering_requested_bytes\":%llu",
        stats_.direct_covering_requested ? "true" : "false",
        (unsigned long long) stats_.direct_covering_batches,
        (unsigned long long) stats_.direct_covering_fallback_batches,
        (unsigned long long) stats_.direct_covering_read_jobs,
        (unsigned long long) stats_.direct_covering_payload_bytes,
        (unsigned long long) stats_.direct_covering_requested_bytes);
    size_t lru_metadata = 0;
    for (const auto & pool : pools_)
        if (pool.lru_index) lru_metadata += pool.lru_index->metadata_allocation_bytes();
    fprintf(file, ",\"gpu_lru_index_enabled\":%s,\"gpu_lru_index_calls\":%llu,"
        "\"gpu_lru_index_candidates\":%llu,\"gpu_lru_bitmap_words\":%llu,"
        "\"gpu_lru_index_rebuilds\":%llu,\"gpu_lru_index_rebuild_ms\":%.3f,"
        "\"gpu_lru_index_metadata_bytes\":%llu",
        gpu_lru_index_enabled_ ? "true" : "false",
        (unsigned long long) gpu_lru_index_calls_, (unsigned long long) gpu_lru_index_candidates_,
        (unsigned long long) gpu_lru_bitmap_words_, (unsigned long long) gpu_lru_index_rebuilds_,
        gpu_lru_index_rebuild_ms_, (unsigned long long) lru_metadata);
    fprintf(file, ",\"ram_replacement\":\"%s\",\"ram_replacement_metadata_bytes\":%llu,"
        "\"ram_s3_small_bytes\":%llu,\"ram_s3_main_bytes\":%llu,\"ram_s3_ghost_bytes\":%llu,"
        "\"ram_s3_small_entries\":%u,\"ram_s3_main_entries\":%u,\"ram_s3_ghost_entries\":%u",
        saved.replacement_policy, (unsigned long long) saved.replacement_metadata_bytes,
        (unsigned long long) saved.s3_small_bytes, (unsigned long long) saved.s3_main_bytes,
        (unsigned long long) saved.s3_ghost_bytes,
        saved.s3_small_entries, saved.s3_main_entries, saved.s3_ghost_entries);
    ssd_read_phase_.append_json(file, direct_);
    fprintf(file, ",\"skip_unused_retain_enabled\":%s,\"unused_retention_decisions_skipped\":%llu",
        skip_unused_retain_requested() ? "true" : "false",
        (unsigned long long) unused_retention_decisions_skipped_);
    if (io_workers_) io_workers_->append_batch_stats(file);
    fputs("}\n", file);
    fclose(file);
    return true;
}

// Release a slot's current occupant (no-op if already empty).
void ExpertCache::activation_lookahead(int current_layer) {
    if (!activation_prefetch_) return;
    const auto started = std::chrono::steady_clock::now();
    if (prediction_future_.valid() &&
        prediction_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        try {
            SchedulerTimer timing(prediction_commit_ms_);
            auto slabs = prediction_future_.get();
            for (auto & slab : slabs) {
                prediction_bytes_ += slab.data.size();
                // Recheck on the owner thread: demand may have loaded the same
                // expert since speculation began. Never publish duplicate tiers.
                if (resident((Role) slab.role, slab.layer, slab.expert) ||
                    host_cache_->contains(ram_key(slab.role, slab.layer, slab.expert))) continue;
                if (host_cache_->put(ram_key(slab.role, slab.layer, slab.expert), slab.data.size(),
                    [&](void * destination) { std::memcpy(destination, slab.data.data(), slab.data.size()); }))
                    prediction_inserted_bytes_ += slab.data.size();
            }
        } catch (const std::exception &) {
            ++prediction_errors_; // Demand execution still uses the authoritative GGUF.
        }
    }
    if (!prediction_future_.valid() && current_layer + 1 < n_layer_ && serves(current_layer + 1)) {
        ++prediction_calls_;
        const int future_layer = current_layer + 1;
        std::vector<int> candidates;
        { SchedulerTimer timing(prediction_rank_ms_); candidates = activation_->predict_next(current_layer, 2); }
        predicted_[future_layer] = candidates;
        prediction_candidates_ += candidates.size();
        struct ReadPlan { int role, layer, expert; std::string path; uint64_t offset; size_t bytes; };
        std::vector<ReadPlan> reads;
        for (int expert : candidates) {
            for (int role = 0; role < N_ROLE; ++role) {
                if (resident((Role) role, future_layer, expert) ||
                    host_cache_->contains(ram_key(role, future_layer, expert))) continue;
                const size_t bytes = tensor((Role) role, future_layer)->nb[2];
                reads.push_back({role, future_layer, expert, fpath_[role][future_layer],
                                 foff_[role][future_layer] + bytes * (uint64_t) expert, bytes});
            }
        }
        if (!reads.empty()) {
            SchedulerTimer timing(prediction_submit_ms_);
            prediction_future_ = std::async(std::launch::async, [reads = std::move(reads)] {
                std::vector<PredictedSlab> slabs;
                for (const auto & read : reads) {
                    PredictedSlab slab{read.role, read.layer, read.expert, std::vector<uint8_t>(read.bytes)};
                    std::ifstream input(read.path, std::ios::binary);
                    input.seekg((std::streamoff) read.offset);
                    if (!input.read((char *) slab.data.data(), (std::streamsize) read.bytes))
                        throw std::runtime_error("Speculative local GGUF read failed");
                    slabs.push_back(std::move(slab));
                }
                return slabs;
            });
        }
    }
    prediction_plan_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

void ExpertCache::note_demands(int layer, const int32_t * ids, int count) {
    if (activation_) { SchedulerTimer timing(activation_observe_ms_); activation_->observe(layer, ids, count); }
    if (activation_scores_)
        for (int role = 0; role < N_ROLE; ++role)
            host_cache_->refresh_priority_range(ram_key(role, layer, 0), n_expert_);
    if (infinity_policy_ && !predicted_[layer].empty()) {
        for (int expert : predicted_[layer])
            if (std::find(ids, ids + count, expert) != ids + count) ++prediction_hits_;
        predicted_[layer].clear();
    }
    if (!admission_) return;
    const bool record_frequency = prefill_cache_policy_.record_decode_frequency();
    for (int index = 0; index < count; ++index) {
        if (ids[index] < 0 || ids[index] >= n_expert_) throw std::out_of_range("Invalid logical expert demand");
        if (record_frequency) admission_->observe((size_t) layer * n_expert_ + ids[index]);
    }
}

bool ExpertCache::can_retain(int layer, int expert) {
    if (!admission_) return true;
    const size_t candidate = (size_t) layer * n_expert_ + expert;
    for (int role = 0; role < N_ROLE; ++role) {
        auto & pool = pools_[layer_pool_[role][layer]];
        if (pool.key2slot.count((int) candidate)) continue;
        const int slot = pick_victim(pool, layer);
        if (slot < 0) return false;
        const int victim = pool.slot2key[slot];
        if (victim >= 0 && (activation_scores_ ? activation_->priority(candidate) <= activation_->priority(victim) :
                                             !admission_->admit(candidate, victim))) return false;
    }
    return true;
}

void ExpertCache::promote_hot_demand(int layer, const int32_t * ids, int count) {
    if (!admission_) return;
    ActiveExpertGuard protected_selection(active_demands_, layer, n_expert_, ids, count);
    int best = -1;
    // Scan all RAM candidates in this rotating layer, not only this token's
    // selected IDs. Otherwise an expert can miss every reconciliation window.
    for (int expert = 0; expert < n_expert_; ++expert) {
        if (!host_resident(layer, expert, false)) continue;
        const auto frequency = admission_->frequency((size_t) layer * n_expert_ + expert);
        if (frequency < 2 || !can_retain(layer, expert)) continue;
        if (best < 0 || frequency > admission_->frequency((size_t) layer * n_expert_ + best)) best = expert;
    }
    if (best >= 0) { ensure_resident(layer, best); ++proactive_promotions_; }
}

void ExpertCache::release_victim(Pool & pool, int slot) {
    const int key = pool.slot2key[slot];
    if (!admission_ || key < 0) { drop_slot(pool, slot); return; }
    const int layer = key / n_expert_;
    // Eviction is at complete-expert granularity. A role miss must not strand
    // the other two immutable slabs in a different managed memory tier.
    for (int role = 0; role < N_ROLE; ++role) {
        auto & other = pools_[layer_pool_[role][layer]];
        const auto found = other.key2slot.find(key);
        if (found != other.key2slot.end()) drop_slot(other, found->second);
    }
}

void ExpertCache::drop_slot(Pool & pool, int slot) {
    // Diagnose stale occupancy BEFORE demotion can change managed RAM ownership.
    const int k = validate_gpu_lru_drop(pool, slot, n_expert_);
    if (k < 0) return;
    demote_to_ram(pool, slot, k); // A failed demotion leaves both indexes untouched.
    drop_gpu_lru_slot(pool, slot, n_expert_);
    l_evict_[k / n_expert_]++;
    stats_.evictions++;
}

void ExpertCache::touch_slot(Pool & pool, int slot) {
    record_gpu_lru_access(pool, slot, clock_);
}

void ExpertCache::rebuild_gpu_lru_indexes() {
    if (!gpu_lru_index_enabled_) return;
    SchedulerTimer timing(gpu_lru_index_rebuild_ms_);
    for (auto & pool : pools_) {
        pool.lru_index->rebuild(pool.n_slots, pool.slot2key, pool.clk);
        ++gpu_lru_index_rebuilds_;
    }
}

void ExpertCache::validate_gpu_lru_indexes() const {
    if (!gpu_lru_index_enabled_) return;
    for (const auto & pool : pools_) {
        if (!pool.lru_index || pool.slot2key.size() != static_cast<size_t>(pool.n_slots) ||
            pool.clk.size() != static_cast<size_t>(pool.n_slots) ||
            !pool.lru_index->invariant(pool.slot2key, pool.clk))
            throw std::logic_error("GPU LRU index disagrees with retained pool clocks/slots");
        size_t occupied = 0;
        for (int slot = 0; slot < pool.n_slots; ++slot) {
            const int key = pool.slot2key[slot];
            if (key < 0) continue;
            const auto found = pool.key2slot.find(key);
            if (found == pool.key2slot.end() || found->second != slot ||
                static_cast<size_t>(key) >= pool.slot_of.size() || pool.slot_of[key] != slot)
                throw std::logic_error("GPU LRU authoritative slot/key/remap maps disagree");
            ++occupied;
        }
        if (occupied != pool.key2slot.size())
            throw std::logic_error("GPU LRU reverse map has an extra key");
    }
}

// Which slot should `layer` take? Free slots first, then LRU. With quotas on,
// a layer that has reached its quota recycles one of its own slots; a layer
// below quota takes the LRU slot of some *over*-quota layer. If no layer is
// over quota (cold pool, or quotas just widened) it degrades to global LRU.
int ExpertCache::pick_victim(Pool & pool, int layer) {
    SchedulerTimer timing(gpu_victim_selection_ms_);
    ++gpu_victim_calls_;
    if (pool.lru_index && !quota_on_ && !activation_scores_) {
        const auto selected = pool.lru_index->pick([&](int slot) {
            const int key = pool.slot2key[slot];
            return !active_demands_.empty() && active_demands_[key];
        });
        ++gpu_lru_index_calls_;
        gpu_lru_index_candidates_ += selected.occupied_examined;
        gpu_lru_bitmap_words_ += selected.bitmap_words_examined;
        return selected.slot;
    }
    const bool self = quota_on_ && pool.quota[layer] >= 0 &&
                      pool.occ[layer] >= pool.quota[layer];
    int victim = -1;
    uint64_t best = UINT64_MAX;
    double lowest_priority = std::numeric_limits<double>::infinity();
    for (int s = 0; s < pool.n_slots; ++s) {
        ++gpu_victim_candidates_;
        const int k = pool.slot2key[s];
        if (k < 0) return s;                     // free slot
        if (!active_demands_.empty() && active_demands_[k]) continue;
        if (quota_on_) {
            const int kl = k / n_expert_;
            if (self) { if (kl != layer) continue; }        // recycle own slot
            else if (pool.occ[kl] <= pool.quota[kl]) continue;  // donor must be over quota
        }
        const double score = activation_scores_ ? activation_->priority(k) : 0;
        if ((activation_scores_ && score < lowest_priority) ||
            ((!activation_scores_ || score == lowest_priority) && pool.clk[s] < best)) {
            lowest_priority = score; best = pool.clk[s]; victim = s;
        }
    }
    if (victim >= 0) return victim;
    for (int s = 0; s < pool.n_slots; ++s) {     // no quota donor: global LRU
        const int key = pool.slot2key[s];
        if (!active_demands_.empty() && key >= 0 && active_demands_[key]) continue;
        if (pool.clk[s] < best) { best = pool.clk[s]; victim = s; }
    }
    return victim;
}

// Fetch (layer,expert) into a free (or LRU-evicted) slot; updates residency maps.
int ExpertCache::install(Pool & pool, int key, int layer, int expert, Role role) {
    const int victim = pick_victim(pool, layer);
    if (victim < 0) throw std::runtime_error("No unprotected GPU cache victim");
    release_victim(pool, victim);
    fetch_slab(role, layer, expert, pool.t, victim);
    publish_gpu_lru_slot(pool, victim, key, layer, clock_);
    return victim;
}

// Claim a slot for `key` (evicting the LRU occupant) without loading data yet.
int ExpertCache::reserve_victim(Pool & pool, int key) {
    const int victim = pick_victim(pool, key / n_expert_);
    if (victim < 0) throw std::runtime_error("No unprotected GPU cache victim");
    release_victim(pool, victim);
    publish_gpu_lru_slot(pool, victim, key, key / n_expert_, clock_);
    return victim;
}

// Read all reserved jobs' slabs from SSD in parallel (each worker uses its own
// file handles). On unified memory the workers pread straight into the VRAM
// slot (zero-copy); otherwise they stage and the slabs are uploaded serially.
void ExpertCache::fetch_parallel(int layer, std::vector<FetchJob> & jobs) {
    const auto io_phase = ssd_read_phase_.begin_fetch(direct_);
    const bool direct_read = io_phase.direct();
    flush_victims();
    // The remote node owns its own disk/RAM reads. Do not read or coalesce its
    // expert payloads on the coordinator before issuing the identifier request.

    const int n = (int) jobs.size();
    if (n == 0) return;
    const auto t0 = std::chrono::steady_clock::now();

    // ---- coalescing: expert slabs of one (role,layer) tensor are contiguous in
    // the GGUF, and a prefill layer's union covers most of them, so scattered
    // 0.6MB preads are merged into big sequential range reads (several times the
    // drive's scattered throughput). Small gaps are read and discarded; small
    // job sets (decode misses, background refill) keep the per-slab path below.
    // Opt-in (QWEN_COALESCE=1): on drives whose sequential read beats their
    // QD-parallel scattered read, merging a layer's union into range reads
    // wins. Measured here on a drive with a degraded tail region (seq 1.9GB/s
    // head / 0.15GB/s tail), 8-thread scattered slab reads extract more from
    // the controller, so scattered stays the default.
    static const bool use_coal = getenv("QWEN_COALESCE") != nullptr;
    static const int  coal_gap = []{ const char * c = getenv("QWEN_COALESCE_GAP");
                                     const int v = c ? atoi(c) : 8; return v < 0 ? 8 : v; }();
    constexpr size_t SEG_BYTES  = 8u << 20;   // one worker read segment
    constexpr size_t SPAN_BYTES = 64u << 20;  // staging cap: runs split at this size
    constexpr int    COAL_MIN   = 16;         // fewer jobs per role: per-slab path

    struct Run { Pool * pool; int role; int e_lo, e_hi; std::vector<const FetchJob *> jobs; };
    std::vector<Run> runs;
    std::vector<FetchJob> singles;

    if (!use_coal || host_cache_) {
        singles = jobs;
    } else {
        for (int r = 0; r < N_ROLE; ++r) {
            std::vector<const FetchJob *> rj;
            for (const auto & j : jobs) if (j.role == r) rj.push_back(&j);
            if (rj.empty()) continue;
            if ((int) rj.size() < COAL_MIN) {
                for (const FetchJob * j : rj) singles.push_back(*j);
                continue;
            }
            std::sort(rj.begin(), rj.end(),
                      [](const FetchJob * a, const FetchJob * b) { return a->expert < b->expert; });
            const size_t nb2 = rj[0]->pool->t->nb[2];
            std::vector<Run> rruns;
            Run cur{ rj[0]->pool, r, rj[0]->expert, rj[0]->expert, { rj[0] } };
            for (size_t i = 1; i < rj.size(); ++i) {
                const int e = rj[i]->expert;
                if (e - cur.e_hi - 1 <= coal_gap &&
                    (size_t) (e - cur.e_lo + 1) * nb2 <= SPAN_BYTES) {
                    cur.e_hi = e;
                    cur.jobs.push_back(rj[i]);
                } else {
                    rruns.push_back(std::move(cur));
                    cur = Run{ rj[i]->pool, r, e, e, { rj[i] } };
                }
            }
            rruns.push_back(std::move(cur));
            // waste guard: a sparse run reads mostly gap bytes — scattered
            // QD-parallel per-slab reads of just the wanted slabs win there
            for (auto & rr : rruns) {
                const int span_slabs = rr.e_hi - rr.e_lo + 1;
                if ((int) rr.jobs.size() * 2 < span_slabs) {
                    for (const FetchJob * j : rr.jobs) singles.push_back(*j);
                } else {
                    runs.push_back(std::move(rr));
                }
            }
        }
    }

    // coalesced runs: workers read disjoint 4KB-aligned segments of the range
    // straight into the (pinned) staging, then the slabs are uploaded — async
    // DMA when pinned (the sync pageable per-slab copies used to dominate).
    // The staging is reused across runs, so a pending async batch is synced
    // before the next run's reads overwrite it.
    for (auto & run : runs) {
        const size_t nb2  = run.pool->t->nb[2];
        const std::string & path = fpath_[run.role][layer];
        const uint64_t base = foff_[run.role][layer] + (uint64_t) run.e_lo * nb2;
        const size_t span = (size_t) (run.e_hi - run.e_lo + 1) * nb2;
        const uint64_t a0    = base & ~4095ull;          // sector-aligned range start
        const size_t   delta = (size_t) (base - a0);
        const size_t   need  = (delta + span + 4095) & ~(size_t) 4095;

        static const bool coal_dbg = getenv("QWEN_COAL_DEBUG") != nullptr;
        const auto tr0 = std::chrono::steady_clock::now();
        if (coal_async_pending_) {   // staging still feeding async H2D: drain first
            ggml_backend_synchronize(backend_);
            coal_async_pending_ = false;
        }
        uint8_t * cbuf = (uint8_t *) coal_host(need);
        static bool coal_dbg_once = false;
        if (coal_dbg && !coal_dbg_once) {
            fprintf(stderr, "coal: staging %s (%zu MB)\n", coal_pinned_ ? "pinned" : "pageable", need >> 20);
            coal_dbg_once = true;
        }

        std::vector<std::pair<size_t, size_t>> segs;   // (offset into range, length)
        for (size_t o = 0; o < need; o += SEG_BYTES)
            segs.push_back({ o, std::min(SEG_BYTES, need - o) });

        std::atomic<size_t> seg_next{0};
        std::atomic<bool>   seg_ok{true};
        auto rworker = [&, direct_read]() {
            FILE * f = nullptr;
#ifdef _WIN32
            void * h = nullptr;
            if (direct_read) { h = win_direct_open(path); if (!h) { seg_ok = false; return; } }
            if (!h)
#endif
            { f = fopen(path.c_str(), "rb"); if (!f) { seg_ok = false; return; } }
            size_t i;
            while ((i = seg_next.fetch_add(1)) < segs.size()) {
                const size_t o = segs[i].first, len = segs[i].second;
#ifdef _WIN32
                if (h) {
                    // aligned offset/len/buffer: read straight into the staging
                    // (a short read is fine iff it still covers the needed span)
                    OVERLAPPED ov{};
                    ov.Offset     = (DWORD) ((a0 + o) & 0xffffffffull);
                    ov.OffsetHigh = (DWORD) ((a0 + o) >> 32);
                    DWORD got = 0;
                    if (!ReadFile((HANDLE) h, cbuf + o, (DWORD) len, &got, &ov) ||
                        (got < len && a0 + o + got < base + span))
                        seg_ok = false;
                    continue;
                }
                if (_fseeki64(f, (long long) (a0 + o), SEEK_SET) != 0 ||
#else
                if (fseeko(f, (off_t) (a0 + o), SEEK_SET) != 0 ||
#endif
                    fread(cbuf + o, 1, len, f) < std::min(len, (size_t) (base + span - (a0 + o))))
                    seg_ok = false;
            }
            if (f) fclose(f);
#ifdef _WIN32
            if (h) CloseHandle((HANDLE) h);
#endif
        };
        const int nth = std::min((int) segs.size(), prefetch_threads_);
        std::vector<std::thread> rts;
        rts.reserve(nth > 0 ? nth - 1 : 0);
        for (int t = 1; t < nth; ++t) rts.emplace_back(rworker);
        rworker();
        for (auto & t : rts) t.join();
        if (!seg_ok) throw std::runtime_error("ExpertCache: coalesced SSD read failed");
        const auto tr1 = std::chrono::steady_clock::now();

        for (const FetchJob * job : run.jobs) {
            const uint8_t * src = cbuf + delta + (size_t) (job->expert - run.e_lo) * nb2;
            if (run.pool->host) {
                memcpy(run.pool->host + (size_t) job->slot * nb2, src, nb2);
            } else if (coal_pinned_) {
                ggml_backend_tensor_set_async(backend_, run.pool->t, src, (size_t) job->slot * nb2, nb2);
                coal_async_pending_ = true;
            } else {
                ggml_backend_tensor_set(run.pool->t, src, (size_t) job->slot * nb2, nb2);
            }
            stats_.fetch_bytes += nb2;
        }
        if (coal_dbg) {
            const auto tr2 = std::chrono::steady_clock::now();
            fprintf(stderr, "coal: run role=%d e=[%d..%d] read %.1fms (%.2f GB/s) upload %.1fms\n",
                    run.role, run.e_lo, run.e_hi,
                    std::chrono::duration<double, std::milli>(tr1 - tr0).count(),
                    need / 1073741824.0 / std::chrono::duration<double>(tr1 - tr0).count(),
                    std::chrono::duration<double, std::milli>(tr2 - tr1).count());
        }
    }
    // pending async uploads need no drain here: later graph compute and slot
    // writes are ordered behind them on the backend stream; the staging itself
    // is protected by the sync-before-reuse above (coal_async_pending_ persists
    // across calls).

    if (singles.empty()) {
        stats_.fetch_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        return;
    }

    // Per-slab scattered reads (QD-parallel across workers) into batches of the
    // pinned staging, uploaded as async DMA — the sync pageable per-slab copies
    // used to cost more than the SSD reads themselves. The staging is drained
    // before each batch reuses it.
    const int ns = (int) singles.size();
    size_t max_nb2 = 0;
    for (const auto & j : singles) max_nb2 = std::max(max_nb2, (size_t) j.pool->t->nb[2]);
    const int batch = std::max<int>(1, (int) ((64u << 20) / max_nb2));

    static const bool request_covering = [] {
        const char * value = getenv("QWEN_DIRECT_PINNED_COVERING");
        return value && std::strcmp(value, "1") == 0;
    }();
    stats_.direct_covering_requested = request_covering;
    static const bool request_pipeline = [] {
        const char * value = getenv("QWEN_DEMAND_IO_PIPELINE");
        return value && std::strcmp(value, "1") == 0;
    }();

    for (int b0 = 0; b0 < ns; b0 += batch) {
        const int bn = std::min(batch, ns - b0);
        if (coal_async_pending_) {   // staging still feeding async H2D: drain first
            ggml_backend_synchronize(backend_);
            coal_async_pending_ = false;
        }
        size_t staging_stride = max_nb2;
        bool use_covering = false;
#ifdef _WIN32
        std::vector<DirectCoveringRead> covering_plans;
        if (request_covering && direct_read && !request_pipeline &&
            std::all_of(singles.begin() + b0, singles.begin() + b0 + bn,
                        [](const FetchJob & job) { return job.pool->host == nullptr; })) {
            size_t candidate_stride = 0;
            if (direct_covering_stride(max_nb2, candidate_stride)) {
                covering_plans.resize(bn);
                use_covering = true;
                for (int j = 0; j < bn; ++j) {
                    const auto & job = singles[b0 + j];
                    const size_t bytes = job.pool->t->nb[2];
                    const uint64_t offset = foff_[job.role][layer] + static_cast<uint64_t>(job.expert) * bytes;
                    if (!direct_covering_range(offset, bytes, covering_plans[j]) ||
                        covering_plans[j].read_bytes > candidate_stride) {
                        use_covering = false;
                        break;
                    }
                }
                if (use_covering) staging_stride = candidate_stride;
            }
        }
#endif
        uint8_t * cbuf = (uint8_t *) coal_host(static_cast<size_t>(bn) * staging_stride);
        // Unsupported modes/layouts explicitly retain the original bounce path.
        if (use_covering && (!coal_pinned_ || (reinterpret_cast<uintptr_t>(cbuf) & 4095))) {
            use_covering = false;
            staging_stride = max_nb2;
        }
        if (use_covering) ++stats_.direct_covering_batches;
        else if (request_covering) ++stats_.direct_covering_fallback_batches;
        std::vector<uint8_t> covering_completed(use_covering ? bn : 0, 0);

        std::atomic<bool> ok{true};
        const int nthreads = std::min(bn, prefetch_threads_);
        // This patch only changes the separate-memory singles path. The old
        // unified-memory and temporary-thread paths retain their behavior.
        const bool pipeline = request_pipeline && io_workers_ &&
            std::all_of(singles.begin() + b0, singles.begin() + b0 + bn,
                        [](const FetchJob & job) { return job.pool->host == nullptr; });
        std::unique_ptr<ExpertReadiness> readiness;
        std::vector<std::shared_ptr<const uint8_t[]>> ram_sources;
        if (pipeline) {
            readiness = std::make_unique<ExpertReadiness>(static_cast<size_t>(bn));
            ++stats_.pipeline_batches;
            stats_.pipeline_jobs += bn;
            ram_sources.resize(bn);
            for (int j = 0; j < bn; ++j) {
                const auto & job = singles[b0 + j];
                const size_t bytes = job.pool->t->nb[2];
                if (host_cache_)
                    ram_sources[j] = host_cache_->transfer_lease(ram_key(job.role, layer, job.expert), bytes,
                        !(readthrough_ram_ || transient_slot(*job.pool, job.slot)));
                if (ram_sources[j]) stats_.pipeline_ram_bytes += bytes;
                else stats_.pipeline_file_bytes += bytes;
            }
        }
        auto read_jobs = [&, direct_read](int tid, ExpertIOFiles & files) {
            auto & tf = files.buffered;
#ifdef __APPLE__
            auto & tfd = files.descriptors;
#endif
#ifdef _WIN32
            auto & th = files.direct;
            auto & tabuf = files.aligned;
#endif
            auto read_one = [&, direct_read](int j) {
                const FetchJob & job = singles[b0 + j];
                const size_t nb2 = job.pool->t->nb[2];
                const std::string & path = fpath_[job.role][layer];
                const size_t off = foff_[job.role][layer] + (size_t) job.expert * nb2;
                uint8_t * range = cbuf + static_cast<size_t>(j) * staging_stride;
                uint8_t * dst = range;
#ifdef _WIN32
                if (use_covering) dst += covering_plans[j].payload_delta;
#endif
                if (pipeline && ram_sources[j]) {
                    std::memcpy(dst, ram_sources[j].get(), nb2);
                    return;
                }
                if (!pipeline && host_cache_ && ((readthrough_ram_ || transient_slot(*job.pool, job.slot)) ? host_cache_->read(ram_key(job.role, layer, job.expert), dst, nb2) : host_cache_->take(ram_key(job.role, layer, job.expert), dst, nb2)))
                    return;
#ifdef _WIN32
                if (direct_read) {
                    void *& hv = th[path];
                    if (!hv) hv = win_direct_open(path);
                    if (use_covering) {
                        if (!hv || !win_direct_read_covering(hv, range, staging_stride, covering_plans[j])) ok = false;
                        else covering_completed[j] = 1; // One writer; owner reads only after worker barrier.
                    } else if (!hv || !win_direct_read(hv, dst, nb2, off, tabuf)) ok = false;
                    return;
                }
#endif
#ifdef __APPLE__
                if (job.pool->host) {   // unified memory: straight into the slot
                    auto it = tfd.find(path);
                    int fd;
                    if (it == tfd.end() || it->second < 0) { fd = open(path.c_str(), O_RDONLY); tfd[path] = fd; }
                    else fd = it->second;
                    if (fd < 0 ||
                        pread(fd, job.pool->host + (size_t) job.slot * nb2, nb2, (off_t) off) != (ssize_t) nb2)
                        ok = false;
                    return;
                }
#endif
                FILE *& f = tf[path];
                if (!f) { f = fopen(path.c_str(), "rb"); if (!f) { ok = false; return; } }
#ifdef _WIN32
                if (_fseeki64(f, (long long) off, SEEK_SET) != 0 ||
#else
                if (fseeko(f, (off_t) off, SEEK_SET) != 0 ||
#endif
                    fread(dst, 1, nb2, f) != nb2) ok = false;
            };
            for (int j = tid; j < bn; j += nthreads) {
                try {
                    read_one(j);
                    if (pipeline) readiness->publish(static_cast<size_t>(j));
                } catch (...) {
                    if (pipeline) readiness->fail(std::current_exception());
                    throw;
                }
            }
        };

        auto upload_one = [&](int j) {
            const FetchJob & job = singles[b0 + j];
            const size_t nb2 = job.pool->t->nb[2];
            stats_.fetch_bytes += nb2;
            if (job.pool->host) return;   // already written in place
            const uint8_t * src = cbuf + static_cast<size_t>(j) * staging_stride;
#ifdef _WIN32
            if (use_covering) src += covering_plans[j].payload_delta;
#endif
            if (weight_audit_ && weight_audit_->active())
                weight_audit_->source(fpath_[job.role][layer],
                    foff_[job.role][layer] + static_cast<uint64_t>(job.expert) * nb2,
                    src, nb2, job.role, job.expert, job.slot);
            remember_ssd_slab(job.role, layer, job.expert, src, nb2, transient_slot(*job.pool, job.slot));
            if (coal_pinned_) {
                ggml_backend_tensor_set_async(backend_, job.pool->t, src, (size_t) job.slot * nb2, nb2);
                coal_async_pending_ = true;
            } else {
                ggml_backend_tensor_set(job.pool->t, src, (size_t) job.slot * nb2, nb2);
            }
        };
        if (pipeline) {
            try {
                io_workers_->run_with_owner_progress(nthreads, read_jobs, [&] {
                    for (int j = 0; j < bn; ++j) {
                        const auto waiting = std::chrono::steady_clock::now();
                        readiness->wait(static_cast<size_t>(j));
                        stats_.pipeline_owner_wait_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - waiting).count();
                        if (!ok) throw std::runtime_error("ExpertCache: pipelined SSD read failed");
                        // Completion acquires the staging writes; no worker
                        // touches this lease again after publishing completion.
                        ram_sources[j].reset();
                        const auto processing = std::chrono::steady_clock::now();
                        upload_one(j);
                        stats_.pipeline_owner_work_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - processing).count();
                    }
                }, bn);
            } catch (...) {
                // run_with_owner_progress already joined all CPU writers.
                // Drain even when a backend enqueue itself threw after issuing
                // work: this staging buffer must not be recycled while in use.
                ggml_backend_synchronize(backend_);
                coal_async_pending_ = false;
                throw;
            }
        } else {
            if (io_workers_) {
                // Every reader completes before the owner uploads from this batch.
                // Workers only read CPU bytes and use the existing host-cache lock.
                io_workers_->run(nthreads, read_jobs, bn);
            } else {
                // Keep the temporary-thread baseline and per-batch file lifetime.
                auto worker = [&](int tid) {
                    ExpertIOFiles files(expert_io_closers());
                    read_jobs(tid, files);
                };
                std::vector<std::thread> ts;
                ts.reserve(nthreads > 0 ? nthreads - 1 : 0);
                for (int t = 1; t < nthreads; ++t) ts.emplace_back(worker, t);
                worker(0);
                for (auto & t : ts) t.join();
            }
            if (!ok) throw std::runtime_error("ExpertCache: parallel SSD read failed");
            for (int j = 0; j < bn; ++j) upload_one(j);
        }
#ifdef _WIN32
        // These are completed direct-file jobs, not RAM hits or physical NAND
        // counters. Requested bytes include alignment covering margins.
        if (use_covering) {
            for (int j = 0; j < bn; ++j) if (covering_completed[j]) {
                ++stats_.direct_covering_read_jobs;
                stats_.direct_covering_payload_bytes += covering_plans[j].payload_bytes;
                stats_.direct_covering_requested_bytes += covering_plans[j].read_bytes;
            }
        }
#endif

    }
    stats_.fetch_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

bool ExpertCache::resident(Role role, int layer, int expert) const {
    if (!serves(layer)) return false;   // another device's layer
    const Pool & pool = pools_[layer_pool_[role][layer]];
    return pool.key2slot.find(layer * n_expert_ + expert) != pool.key2slot.end();
}

const int32_t * ExpertCache::slot_of_row(Role role, int layer) const {
    const Pool & pool = pools_[layer_pool_[role][layer]];
    return &pool.slot_of[(size_t) layer * n_expert_];
}

void ExpertCache::touch(Role role, int layer, int expert) {
    if (!serves(layer)) return;
    Pool & pool = pools_[layer_pool_[role][layer]];
    const int key = layer * n_expert_ + expert;
    auto it = pool.key2slot.find(key);
    if (it == pool.key2slot.end()) return;
    pool.count[key]++;
    touch_slot(pool, it->second);
    stats_.hits++;
    l_hits_[layer]++;
}

int ExpertCache::slot_for(Pool & pool, int layer, int expert, Role role) {
    const int key = layer * n_expert_ + expert;
    pool.count[key]++;

    auto it = pool.key2slot.find(key);
    if (it != pool.key2slot.end()) {
        stats_.hits++;
        l_hits_[layer]++;
        touch_slot(pool, it->second);
        return it->second;
    }
    stats_.misses++;
    l_misses_[layer]++;
    return install(pool, key, layer, expert, role);
}

// Profile file format (text):
//   QWENEXPCACHE 1
//   <n_layer> <n_expert>
//   <role> <layer> <expert> <count>   (one per line, hot experts)
bool ExpertCache::save_profile(const std::string & path) const {
    validate_gpu_lru_indexes();
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fprintf(f, "QWENEXPCACHE 1\n%d %d\n", n_layer_, n_expert_);
    for (const auto & pool : pools_) {
        for (const auto & kv : pool.count) {
            const int layer  = kv.first / n_expert_;
            const int expert = kv.first % n_expert_;
            fprintf(f, "%d %d %d %llu\n", pool.role, layer, expert,
                    (unsigned long long) kv.second);
        }
    }
    fclose(f);
    return true;
}

size_t ExpertCache::load_prefetch(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return 0;

    int ver = 0, nl = 0, ne = 0;
    if (fscanf(f, "QWENEXPCACHE %d", &ver) != 1 || ver != 1 ||
        fscanf(f, "%d %d", &nl, &ne) != 2 || nl != n_layer_ || ne != n_expert_) {
        fclose(f);
        return 0;   // missing / incompatible profile -> cold start
    }

    struct Ent { int role, layer, expert; uint64_t count; };
    std::vector<Ent> ents;
    int r, l, e; unsigned long long c;
    while (fscanf(f, "%d %d %d %llu", &r, &l, &e, &c) == 4) {
        if (r >= 0 && r < N_ROLE && l >= 0 && l < n_layer_ && e >= 0 && e < n_expert_ && serves(l))
            ents.push_back({ r, l, e, c });   // a profile covers the whole model;
                                              // take only this device's layers
    }
    fclose(f);

    // Highest-frequency experts first; fill each pool up to its capacity.
    std::sort(ents.begin(), ents.end(),
              [](const Ent & a, const Ent & b) { return a.count > b.count; });

    size_t n_pref = 0;

    for (const auto & en : ents) {
        Pool & pool = pools_[layer_pool_[en.role][en.layer]];
        if ((int) pool.key2slot.size() >= pool.n_slots) continue;   // pool full
        const int key = en.layer * n_expert_ + en.expert;
        if (pool.key2slot.count(key)) continue;                     // already resident
        install(pool, key, en.layer, en.expert, (Role) en.role);
        n_pref++;
    }
    return n_pref;
}

void ExpertCache::note_want(int layer, bool absent) {
    l_want_[layer]++;
    if (absent) l_wmiss_[layer]++;
}

void ExpertCache::reset_layer_stats() {
    std::fill(l_hits_.begin(),   l_hits_.end(),   0);
    std::fill(l_misses_.begin(), l_misses_.end(), 0);
    std::fill(l_evict_.begin(),  l_evict_.end(),  0);
    std::fill(l_want_.begin(),   l_want_.end(),   0);
    std::fill(l_wmiss_.begin(),  l_wmiss_.end(),  0);
}

void ExpertCache::layer_stats(std::vector<LayerStat> & out) const {
    out.assign(n_layer_, LayerStat{});
    for (int il = 0; il < n_layer_; ++il) {
        if (!serves(il)) continue;      // left zeroed: another device reports it
        LayerStat & ls = out[il];
        ls.hits      = l_hits_[il];
        ls.misses    = l_misses_[il];
        ls.evictions = l_evict_[il];
        ls.want      = l_want_[il];
        ls.want_miss = l_wmiss_[il];
        // "resident" = usable by the router: present in all three role pools.
        int n = 0;
        for (int e = 0; e < n_expert_; ++e)
            n += resident(GATE, il, e) && resident(UP, il, e) && resident(DOWN, il, e);
        ls.resident = n;
    }
}

// Per-layer view of the shared pools. `resident` is the layer's share of the
// slots right now; hit/miss/evict are cumulative since the last reset.
void ExpertCache::dump_layer_stats(const char * tag) const {
    std::vector<LayerStat> ls;
    layer_stats(ls);
    int total_slots = 0;
    for (const auto & p : pools_) total_slots += p.n_slots;
    fprintf(stderr, "\n---- per-layer expert cache [%s] (%d layers x %d experts, "
                    "%d slots over %zu pools) ----\n",
            tag, n_layer_, n_expert_, total_slots, pools_.size());
    fprintf(stderr, "layer  resident   share%%      acc     hit%%    misses   evict"
                    "   want  wantmiss%%\n");
    uint64_t th = 0, tm = 0, te = 0, tw = 0, twm = 0; int tr = 0;
    for (int il = 0; il < n_layer_; ++il) {
        if (!serves(il)) continue;
        const LayerStat & s = ls[il];
        const uint64_t acc = s.hits + s.misses;
        fprintf(stderr, "%5d  %8d  %6.2f  %9llu  %6.1f  %8llu  %6llu  %6llu  %8.1f\n",
                il, s.resident,
                100.0 * (double) s.resident / (double) n_expert_,
                (unsigned long long) acc,
                acc ? 100.0 * (double) s.hits / (double) acc : 0.0,
                (unsigned long long) s.misses,
                (unsigned long long) s.evictions,
                (unsigned long long) s.want,
                s.want ? 100.0 * (double) s.want_miss / (double) s.want : 0.0);
        tr += s.resident; th += s.hits; tm += s.misses; te += s.evictions;
        tw += s.want; twm += s.want_miss;
    }
    const uint64_t ta = th + tm;
    fprintf(stderr, "  all  %8d  %6.2f  %9llu  %6.1f  %8llu  %6llu  %6llu  %8.1f\n",
            tr, 100.0 * (double) tr / (double) (n_layer_ * n_expert_),
            (unsigned long long) ta, ta ? 100.0 * (double) th / (double) ta : 0.0,
            (unsigned long long) tm, (unsigned long long) te,
            (unsigned long long) tw, tw ? 100.0 * (double) twm / (double) tw : 0.0);
    fprintf(stderr, "----\n\n");
}

void ExpertCache::ensure_resident(int layer, int expert) {
    if (!serves(layer)) return;
    if (ssd_) {
        // Preserve residency bookkeeping and batch all three role misses locally or remotely.
        int32_t gate, up, down;
        const int32_t expert_id = expert;
        ensure(layer, &expert_id, 1, &gate, &up, &down, false);
        return;
    }
    for (int r = 0; r < N_ROLE; ++r) {
        Pool & pool = pools_[layer_pool_[r][layer]];
        const int key = layer * n_expert_ + expert;
        if (pool.key2slot.find(key) == pool.key2slot.end()) {
            stats_.misses++;
            l_misses_[layer]++;
            install(pool, key, layer, expert, (Role) r);
        }
    }
}

void ExpertCache::ensure(int layer, const int32_t * expert_ids, int n,
                         int32_t * slot_gate, int32_t * slot_up, int32_t * slot_down, bool record_demand) {
    // Routing bug rather than a runtime condition: fail loudly instead of
    // indexing pools_ with the -1 that marks another device's layer.
    if (!serves(layer))
        throw std::runtime_error("ExpertCache::ensure on layer " + std::to_string(layer) +
                                 " which this device does not serve");
    if (record_demand) note_demands(layer, expert_ids, n);
    const bool audit_group = weight_audit_ && weight_audit_->begin(layer, n, n_used_);
    ActiveExpertGuard protected_selection(active_demands_, layer, n_expert_, expert_ids, n);
    std::unordered_map<int, int> temporary[N_ROLE];
    Pool * pools3[N_ROLE] = {
        &pools_[layer_pool_[GATE][layer]],
        &pools_[layer_pool_[UP][layer]],
        &pools_[layer_pool_[DOWN][layer]],
    };
    int32_t * outs[N_ROLE] = { slot_gate, slot_up, slot_down };

    // RAM source: sequential is fine (memcpy is fast).
    if (!ssd_) {
        for (int i = 0; i < n; ++i) {
            const int e = (int) expert_ids[i];
            for (int r = 0; r < N_ROLE; ++r)
                outs[r][i] = slot_for(*pools3[r], layer, e, (Role) r);
        }
        return;
    }

    // SSD source: resolve hits, reserve victims for misses, then read the misses
    // in parallel (overlapping disk latency) before uploading them.
    std::vector<FetchJob> jobs;
    const bool skip_unused_retain = skip_unused_retain_requested();
    for (int i = 0; i < n; ++i) {
        const int e = (int) expert_ids[i];
        const int key = layer * n_expert_ + e;
        // Covered roles cannot install/evict in the unchanged loop below.
        // A new installation still makes its decision at the original point.
        const bool needs_install = !skip_unused_retain ||
            expert_role_needs_install(key, pools3, temporary);
        if (skip_unused_retain && !needs_install) ++unused_retention_decisions_skipped_;
        const bool retain = needs_install && can_retain(layer, e);
        for (int r = 0; r < N_ROLE; ++r) {
            Pool & pool = *pools3[r];
            pool.count[key]++;
            auto it = pool.key2slot.find(key);
            if (it != pool.key2slot.end()) {
                stats_.hits++;
                l_hits_[layer]++;
                touch_slot(pool, it->second);
                outs[r][i] = it->second;
            } else {
                auto repeated = temporary[r].find(key);
                if (repeated != temporary[r].end()) {
                    outs[r][i] = repeated->second;
                    stats_.hits++;
                    l_hits_[layer]++;
                    continue;
                }
                stats_.misses++;
                l_misses_[layer]++;
                int slot;
                if (retain) {
                    slot = reserve_victim(pool, key);
                    ++retained_role_fills_;
                } else {
                    if ((int) temporary[r].size() >= pool.scratch_slots)
                        throw std::runtime_error("Transient expert workspace exhausted");
                    slot = pool.n_slots + (int) temporary[r].size();
                    temporary[r][key] = slot;
                    ++transient_role_fills_;
                }
                outs[r][i] = slot;
                jobs.push_back({ &pool, r, e, slot });
            }
        }
    }
    fetch_parallel(layer, jobs);
    if (audit_group) {
        if (weight_audit_->wants_gpu()) {
            // Diagnostic barrier, never a performance configuration. Include
            // retained GPU hits as well as freshly uploaded slabs.
            ggml_backend_synchronize(backend_);
            for (int index = 0; index < n; ++index) {
                const int expert = expert_ids[index];
                for (int role = 0; role < N_ROLE; ++role) {
                    auto * tensor = pools3[role]->t;
                    const size_t bytes = tensor->nb[2];
                    const int slot = outs[role][index];
                    weight_audit_->destination(fpath_[role][layer],
                        foff_[role][layer] + static_cast<uint64_t>(expert) * bytes,
                        bytes, role, expert, slot, [&](void * destination, size_t count) {
                            ggml_backend_tensor_get(tensor, destination, static_cast<size_t>(slot) * bytes, count);
                        });
                }
            }
        }
        weight_audit_->end();
    }
}

} // namespace questwend
