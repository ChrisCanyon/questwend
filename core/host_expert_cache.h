#pragma once
#include "scheduler_timing.h"
#include "ram_s3fifo_metadata.h"

// Bounded victim cache for immutable expert slabs evicted from GPU memory.
// A successful take removes its RAM copy: promotion does not permanently keep
// duplicate weights in both managed tiers. The owner scopes keys to one model.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <new>
#include <mutex>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <vector>
#include <set>
#include <tuple>

namespace questwend {

class HostExpertCache {
public:
    struct Stats {
        uint64_t capacity_bytes = 0, resident_bytes = 0, peak_bytes = 0;
        uint64_t hits = 0, misses = 0, hit_bytes = 0;
        uint64_t admissions = 0, admitted_bytes = 0;
        uint64_t evictions = 0, evicted_bytes = 0, rejected = 0;
        uint64_t cpu_reads = 0, cpu_read_bytes = 0;
        uint64_t pooled_bytes = 0, allocations = 0, reused_allocations = 0;
        double allocation_ms = 0, fill_ms = 0, cpu_copy_ms = 0;
        uint64_t borrowed_reads = 0, borrowed_bytes = 0;
        uint64_t retired_leased_bytes = 0;
        uint64_t eviction_candidates = 0;
        double eviction_selection_ms = 0;
        double priority_refresh_ms = 0;
        uint64_t priority_refresh_entries = 0;
        const char * replacement_policy = "lru";
        uint64_t replacement_metadata_bytes = 0;
        uint64_t s3_small_bytes = 0, s3_main_bytes = 0, s3_ghost_bytes = 0;
        uint32_t s3_small_entries = 0, s3_main_entries = 0, s3_ghost_entries = 0;
    };

    explicit HostExpertCache(size_t capacity) { stats_.capacity_bytes = capacity; }

    // Configuration occurs before the first payload admission. Dense keys and
    // a fixed metadata allocation bound memory and avoid allocating on hits.
    void enable_s3fifo_role_slabs(uint32_t key_count, size_t maximum_role_bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (priority_) throw std::invalid_argument("S3FIFO RAM replacement conflicts with activation priority");
        if (s3_ || !entries_.empty() || retired_bytes_ || stats_.pooled_bytes)
            throw std::invalid_argument("Configure S3FIFO before admitting RAM weights");
        const uint64_t small_capacity = static_cast<uint64_t>(stats_.capacity_bytes * 0.10);
        if (!maximum_role_bytes || maximum_role_bytes >= small_capacity)
            throw std::invalid_argument("S3FIFO requires every role slab smaller than its 10% small queue");
        auto metadata = std::make_unique<RamS3FifoMetadata>(key_count, stats_.capacity_bytes);
        s3_key_count_ = key_count;
        s3_maximum_role_bytes_ = maximum_role_bytes;
        s3_ = std::move(metadata);
    }

    void set_priority(std::function<double(uint64_t)> score, bool indexed = false) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (s3_ && score) throw std::invalid_argument("Activation priority conflicts with S3FIFO RAM replacement");
        priority_ = std::move(score);
        indexed_priority_ = indexed && (bool) priority_;
        ranked_.clear();
        if (indexed_priority_)
            for (auto & item : entries_) refresh_rank(item.first, item.second);
    }

    // Scores change for one observed layer. Update only its resident keys;
    // entries_ is ordered by the role/layer/expert key, so this skips other layers.
    void refresh_priority_range(uint64_t first, uint64_t count) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!indexed_priority_) return;
        SchedulerTimer timing(stats_.priority_refresh_ms);
        for (auto found = entries_.lower_bound(first); found != entries_.end() && found->first - first < count; ++found) {
            refresh_rank(found->first, found->second);
            ++stats_.priority_refresh_entries;
        }
    }

    bool take(uint64_t key, void * destination, size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) { stats_.misses++; return false; }
        if (found->second.bytes != bytes)
            throw std::runtime_error("RAM expert cache: key reused with another slab size");
        std::memcpy(destination, found->second.data.get(), bytes);
        stats_.hits++;
        stats_.hit_bytes += bytes;
        erase(found);
        return true;
    }

    // Read-through hits preserve immutable RAM weights across GPU residency.
    bool read(uint64_t key, void * destination, size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) { stats_.misses++; return false; }
        if (found->second.bytes != bytes)
            throw std::runtime_error("RAM expert cache: read size mismatch");
        std::memcpy(destination, found->second.data.get(), bytes);
        touch(found);
        stats_.hits++;
        stats_.hit_bytes += bytes;
        return true;
    }

    bool contains_and_touch(uint64_t key) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) return false;
        // Eligibility inspection preserves default LRU recency, but is not a
        // successful weight read and must not manufacture S3FIFO frequency.
        touch(found, false);
        return true;
    }

    bool contains(uint64_t key) const {
        std::lock_guard<std::mutex> guard(mutex_);
        return entries_.find(key) != entries_.end();
    }

    // Read a consecutive layer/role row under one lock for dashboard telemetry.
    // It does not touch replacement order or frequency counters.
    std::vector<uint8_t> presence_range(uint64_t first, size_t count) const {
        std::vector<uint8_t> present(count, 0);
        std::lock_guard<std::mutex> guard(mutex_);
        for (auto item = entries_.lower_bound(first);
             item != entries_.end() && item->first - first < count; ++item)
            present[static_cast<size_t>(item->first - first)] = 1;
        return present;
    }

    // Owner-only source classification for an I/O batch. The immutable lease
    // freezes bytes before concurrent owner admissions can evict their key.
    // Taking removes managed RAM ownership immediately; the existing retired
    // lease accounting prevents recycling or exceeding the cache's byte cap.
    std::shared_ptr<const uint8_t[]> transfer_lease(uint64_t key, size_t bytes, bool remove) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) { stats_.misses++; return {}; }
        if (found->second.bytes != bytes)
            throw std::runtime_error("RAM expert transfer lease size mismatch");
        std::shared_ptr<const uint8_t[]> result = found->second.data;
        if (remove) erase(found);
        else touch(found);
        stats_.hits++;
        stats_.hit_bytes += bytes;
        return result;
    }

    // A CPU graph may read an immutable weight in place. Holding this shared
    // owner prevents eviction/recycling from invalidating or changing its bytes.
    std::shared_ptr<const uint8_t[]> borrow(uint64_t key, size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) return {};
        if (found->second.bytes != bytes) throw std::runtime_error("RAM expert borrow size mismatch");
        touch(found);
        stats_.borrowed_reads++;
        stats_.borrowed_bytes += bytes;
        return found->second.data;
    }

    // CPU execution reads a cached weight without promoting or removing it.
    bool copy(uint64_t key, void * destination, size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) return false;
        if (found->second.bytes != bytes)
            throw std::runtime_error("RAM expert cache: CPU read size mismatch");
        const auto started = std::chrono::steady_clock::now();
        std::memcpy(destination, found->second.data.get(), bytes);
        stats_.cpu_copy_ms += elapsed(started);
        touch(found);
        stats_.cpu_reads++;
        stats_.cpu_read_bytes += bytes;
        return true;
    }

    // read fills a private, uninitialized allocation before it becomes visible.
    // Exceptions never publish a partially filled entry.
    bool put(uint64_t key, size_t bytes, const std::function<void(void *)> & read) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!bytes || bytes > stats_.capacity_bytes) { stats_.rejected++; return false; }
        if (s3_ && (key >= s3_key_count_ || bytes > s3_maximum_role_bytes_))
            throw std::invalid_argument("RAM role slab exceeds configured S3FIFO key/size bounds");
        auto previous = entries_.find(key);
        if (previous != entries_.end()) erase(previous);
        // Ghost intent is captured before capacity evictions, as in the source
        // S3FIFO implementation. Failed admissions may discard ghost history;
        // resident payload ownership and physical byte limits remain authoritative.
        const auto intent = s3_ ? s3_->prepare_admission(static_cast<uint32_t>(key), bytes)
                                : RamS3FifoMetadata::Admission{};
        if (s3_ && !intent.eligible) { stats_.rejected++; return false; }
        reap_retired();
        while (stats_.resident_bytes + retired_bytes_ > stats_.capacity_bytes - bytes) {
            // Evicting leased data does not free its allocation until the CPU
            // finishes. Reject admission when those live leases occupy the cap.
            if (order_.empty()) { stats_.rejected++; return false; }
            evict_oldest();
        }
        // Keep a small pool of released slabs. Repeated VirtualAlloc/page faults
        // for megabyte weights otherwise dominate admission on Windows. Both
        // live entries and this pool count against the same capacity bound.
        const auto allocating = std::chrono::steady_clock::now();
        std::shared_ptr<uint8_t[]> data;
        auto spare = spare_.find(bytes);
        if (spare != spare_.end() && !spare->second.empty()) {
            data = std::move(spare->second.back());
            spare->second.pop_back();
            stats_.pooled_bytes -= bytes;
            stats_.reused_allocations++;
        } else {
            while (stats_.pooled_bytes > stats_.capacity_bytes - stats_.resident_bytes - retired_bytes_ - bytes)
                free_spare();
            data.reset(new uint8_t[bytes], std::default_delete<uint8_t[]>());
            stats_.allocations++;
        }
        stats_.allocation_ms += elapsed(allocating);
        const auto filling = std::chrono::steady_clock::now();
        read(data.get());
        stats_.fill_ms += elapsed(filling);
        // Publish list, payload map and optional policy as one transaction.
        // Any allocation failure removes only this new publication. Previously
        // completed capacity evictions and nonauthoritative ghost rotations stay.
        auto new_order = order_.end();
        auto new_entry = entries_.end();
        try {
#ifdef QWEN_CACHE_TEST_FAULTS
            maybe_fail_publish_allocation(1);
#endif
            order_.push_back(key);
            new_order = std::prev(order_.end());
#ifdef QWEN_CACHE_TEST_FAULTS
            maybe_fail_publish_allocation(2);
#endif
            const auto inserted = entries_.emplace(key,
                Entry{std::move(data), bytes, new_order, 0, ++touch_clock_});
            if (!inserted.second) throw std::logic_error("RAM expert key already published");
            new_entry = inserted.first;
            if (indexed_priority_) {
#ifdef QWEN_CACHE_TEST_FAULTS
                maybe_fail_publish_allocation(3);
#endif
                refresh_rank(key, new_entry->second);
            }
            if (s3_ && !s3_->commit_admission(static_cast<uint32_t>(key), bytes, intent))
                throw std::logic_error("RAM S3FIFO declined a validated role slab");
        } catch (...) {
            if (s3_) s3_->remove_external(static_cast<uint32_t>(key));
            if (new_entry != entries_.end()) {
                if (indexed_priority_) ranked_.erase(std::make_tuple(
                    new_entry->second.score, new_entry->second.touched_at, key));
                entries_.erase(new_entry);
            }
            if (new_order != order_.end()) order_.erase(new_order);
            throw;
        }
        stats_.resident_bytes += bytes;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.resident_bytes);
        stats_.admissions++;
        stats_.admitted_bytes += bytes;
        return true;
    }

    void trim(size_t maximum_resident_bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        spare_.clear();
        stats_.pooled_bytes = 0;
        while (stats_.resident_bytes > maximum_resident_bytes) evict_oldest(false);
    }

    Stats stats() const {
        std::lock_guard<std::mutex> guard(mutex_);
        auto result = stats_;
        if (s3_) {
            const auto queues = s3_->snapshot();
            result.replacement_policy = "s3fifo-role-slabs";
            result.replacement_metadata_bytes = s3_->metadata_allocation_bytes();
            result.s3_small_bytes = queues.small_bytes;
            result.s3_main_bytes = queues.main_bytes;
            result.s3_ghost_bytes = queues.ghost_bytes;
            result.s3_small_entries = queues.small_entries;
            result.s3_main_entries = queues.main_entries;
            result.s3_ghost_entries = queues.ghost_entries;
        }
        for (const auto & retired : retired_)
            if (!retired.data.expired()) result.retired_leased_bytes += retired.bytes;
        return result;
    }

#ifdef QWEN_CACHE_TEST_FAULTS
    // Native-unit-only deterministic seams: 1=retirement record,
    // 2=spare bucket, 3=spare vector growth. Absent from runtime builds.
    void fail_erase_allocation_for_test(int point) { erase_allocation_fault_ = point; }
    void fail_publish_allocation_for_test(int point) { publish_allocation_fault_ = point; }
    uint64_t s3_frequency_for_test(uint32_t key) const {
        std::lock_guard<std::mutex> guard(mutex_);
        return s3_ ? s3_->frequency(key) : 0;
    }
    uint64_t s3_digest_for_test() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return s3_ ? s3_->debug_digest() : 0;
    }
    bool invariant_for_test() const {
        std::lock_guard<std::mutex> guard(mutex_);
        uint64_t bytes = 0;
        if (order_.size() != entries_.size()) return false;
        for (const auto key : order_) if (!entries_.count(key)) return false;
        for (const auto & item : entries_) {
            bytes += item.second.bytes;
            if (s3_ && !s3_->resident(static_cast<uint32_t>(item.first))) return false;
        }
        return bytes == stats_.resident_bytes &&
            (!s3_ || (s3_->invariant() && s3_->resident_bytes() == bytes));
    }
#endif

private:
#ifdef QWEN_CACHE_TEST_FAULTS
    int erase_allocation_fault_ = 0, publish_allocation_fault_ = 0;
    void maybe_fail_publish_allocation(int point) {
        if (publish_allocation_fault_ == point) {
            publish_allocation_fault_ = 0;
            throw std::bad_alloc();
        }
    }
    void maybe_fail_erase_allocation(int point) {
        if (erase_allocation_fault_ == point) {
            erase_allocation_fault_ = 0;
            throw std::bad_alloc();
        }
    }
#endif
    struct Entry {
        std::shared_ptr<uint8_t[]> data;
        size_t bytes;
        std::list<uint64_t>::iterator order;
        double score = 0;
        uint64_t touched_at = 0;
    };
    using Entries = std::map<uint64_t, Entry>;

    void refresh_rank(uint64_t key, Entry & entry) {
        ranked_.erase(std::make_tuple(entry.score, entry.touched_at, key));
        entry.score = priority_(key);
        ranked_.emplace(entry.score, entry.touched_at, key);
    }

    void touch(Entries::iterator found, bool successful_read = true) {
        auto & entry = found->second;
        if (s3_ && successful_read) s3_->reused(static_cast<uint32_t>(found->first));
        if (indexed_priority_) ranked_.erase(std::make_tuple(entry.score, entry.touched_at, found->first));
        entry.touched_at = ++touch_clock_;
        if (indexed_priority_) refresh_rank(found->first, entry);
        order_.splice(order_.end(), order_, entry.order);
    }

    struct RetiredSlab { std::weak_ptr<uint8_t[]> data; size_t bytes; };

    void reap_retired() {
        retired_.erase(std::remove_if(retired_.begin(), retired_.end(), [&](const auto & slab) {
            if (!slab.data.expired()) return false;
            retired_bytes_ -= slab.bytes;
            return true;
        }), retired_.end());
    }

    static double elapsed(std::chrono::steady_clock::time_point started) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }

    void free_spare() {
        for (auto & entry : spare_) {
            if (entry.second.empty()) continue;
            entry.second.pop_back();
            stats_.pooled_bytes -= entry.first;
            return;
        }
        throw std::logic_error("RAM expert spare accounting mismatch");
    }

    void erase(Entries::iterator found, bool recycle = true, bool capacity_eviction = false) {
        if (s3_ && (!s3_->resident(static_cast<uint32_t>(found->first)) ||
            (capacity_eviction && !s3_->pending_victim_is(static_cast<uint32_t>(found->first)))))
            throw std::logic_error("RAM S3FIFO payload/metadata ownership mismatch");
        const size_t bytes = found->second.bytes;
        const bool leased = found->second.data.use_count() > 1;
        bool pooled = false;
        // Retirement metadata is required to keep live leases within the byte
        // cap. Allocate it BEFORE changing the rank, residency or key maps.
        // A bad_alloc therefore leaves the original entry fully authoritative.
        if (leased) {
#ifdef QWEN_CACHE_TEST_FAULTS
            maybe_fail_erase_allocation(1);
#endif
            retired_.push_back({found->second.data, bytes});
        } else {
            constexpr size_t MAX_SPARE_BYTES = 64ull << 20;
            if (recycle && bytes <= MAX_SPARE_BYTES - stats_.pooled_bytes) {
                // Spare reuse is optional. If either bucket or vector growth
                // fails, complete the eviction by discarding the unleased data.
                // Keep the entry's shared owner until pooling has succeeded.
                try {
#ifdef QWEN_CACHE_TEST_FAULTS
                    maybe_fail_erase_allocation(2);
#endif
                    auto bucket = spare_.try_emplace(bytes).first;
#ifdef QWEN_CACHE_TEST_FAULTS
                    maybe_fail_erase_allocation(3);
#endif
                    bucket->second.push_back(found->second.data);
                    pooled = true;
                } catch (const std::bad_alloc &) {
                    pooled = false;
                }
            }
        }
        // Metadata has been validated under this same owner mutex. Its commit
        // does not allocate, and only follows successful lease retirement.
        if (s3_) {
            if (capacity_eviction) s3_->commit_capacity_eviction(static_cast<uint32_t>(found->first));
            else s3_->remove_external(static_cast<uint32_t>(found->first));
        }
        // All required fallible preparation is complete. The remaining tuple
        // lookup, scalar accounting and erasures do not allocate.
        if (indexed_priority_)
            ranked_.erase(std::make_tuple(found->second.score, found->second.touched_at, found->first));
        stats_.resident_bytes -= bytes;
        if (leased) retired_bytes_ += bytes;
        if (pooled) stats_.pooled_bytes += bytes;
        order_.erase(found->second.order);
        entries_.erase(found);
    }

    void evict_oldest(bool recycle = true) {
        if (order_.empty()) throw std::logic_error("RAM expert cache accounting mismatch");
        auto selected = order_.begin();
        {
        SchedulerTimer timing(stats_.eviction_selection_ms);
        if (s3_) {
            const auto key = s3_->select_victim();
            const auto found = entries_.find(key);
            if (found == entries_.end()) {
                s3_->cancel_victim();
                throw std::logic_error("RAM S3FIFO selected a missing payload");
            }
            selected = found->second.order;
        } else if (indexed_priority_) {
            if (ranked_.empty()) throw std::logic_error("RAM eviction priority index is empty");
            selected = entries_.at(std::get<2>(*ranked_.begin())).order;
        } else if (priority_) {
            double lowest = priority_(*selected);
            for (auto entry = std::next(selected); entry != order_.end(); ++entry) {
                ++stats_.eviction_candidates;
                const double score = priority_(*entry);
                if (score < lowest) { lowest = score; selected = entry; }
            }
        }
        }
        const auto found = entries_.find(*selected);
        const size_t bytes = found->second.bytes;
        try {
            erase(found, recycle, true);
        } catch (...) {
            if (s3_) s3_->cancel_victim();
            throw;
        }
        stats_.evictions++;
        stats_.evicted_bytes += bytes;
    }

    mutable std::mutex mutex_;
    std::list<uint64_t> order_;
    Entries entries_;
    std::map<size_t, std::vector<std::shared_ptr<uint8_t[]>>> spare_;
    std::vector<RetiredSlab> retired_;
    size_t retired_bytes_ = 0;
    std::function<double(uint64_t)> priority_;
    bool indexed_priority_ = false;
    uint64_t touch_clock_ = 0;
    std::set<std::tuple<double, uint64_t, uint64_t>> ranked_;
    Stats stats_;
    std::unique_ptr<RamS3FifoMetadata> s3_;
    uint32_t s3_key_count_ = 0;
    size_t s3_maximum_role_bytes_ = 0;
};

} // namespace questwend
