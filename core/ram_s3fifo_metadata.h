// SPDX-License-Identifier: Apache-2.0
#pragma once

// Role-slab metadata ONLY. No weights, shared owners, CUDA, file I/O or host
// cache mutation. Derived from libCacheSim S3FIFO.c/FIFO.c at commit
//93529a1faef2c57bbdb9323213b78cad2adec1cd, Juncheng Yang and contributors,
// Apache-2.0 (see adjacent LICENSE/PROVENANCE). Current2024 warm-fill variant.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace questwend {

class RamS3FifoMetadata {
public:
    static constexpr uint32_t none = std::numeric_limits<uint32_t>::max();
    enum class Queue : uint8_t { absent, small_queue, main, ghost };
    struct Admission { bool eligible = false, ghost_hit = false; };
    struct Snapshot {
        uint64_t small_bytes, main_bytes, ghost_bytes;
        uint32_t small_entries, main_entries, ghost_entries;
        bool has_evicted;
    };

    RamS3FifoMetadata(uint32_t key_count, uint64_t capacity)
        : nodes_(checked_key_count(key_count,capacity)), capacity_(capacity), small_capacity_(static_cast<uint64_t>(capacity*0.10)),
          main_capacity_(capacity-small_capacity_), ghost_capacity_(static_cast<uint64_t>(capacity*0.90)) {
    }
    RamS3FifoMetadata(const RamS3FifoMetadata &) = delete;
    RamS3FifoMetadata & operator=(const RamS3FifoMetadata &) = delete;

    uint64_t resident_bytes() const noexcept { return queues_[1].bytes + queues_[2].bytes; }
    size_t metadata_allocation_bytes() const noexcept { return nodes_.capacity()*sizeof(Node); }
    bool resident(uint32_t key) const noexcept {
        return key < nodes_.size() && (nodes_[key].queue == Queue::small_queue || nodes_[key].queue == Queue::main);
    }
    Queue queue_of(uint32_t key) const noexcept { return key < nodes_.size() ? nodes_[key].queue : Queue::absent; }
    uint64_t frequency(uint32_t key) const noexcept { return key < nodes_.size() ? nodes_[key].frequency : 0; }

    // Capture ghost re-admission BEFORE space-making, matching S3FIFO_find:
    // unlink ghost metadata now, so it cannot consume ghost capacity while
    // eviction inserts new ghosts. This never removes a resident or payload.
    // An aborted admission may discard this nonauthoritative ghost history.
    Admission prepare_admission(uint32_t key, uint64_t bytes) noexcept {
        if (key >= nodes_.size() || resident(key) || !bytes)
            return {};
        const bool ghost_hit=nodes_[key].queue == Queue::ghost;
        if (ghost_hit) unlink(key);
        return {bytes <= small_capacity_,ghost_hit};
    }

    // Actual read/copy/borrow hit, not a contains()/residency inspection.
    bool reused(uint32_t key) noexcept {
        if (!resident(key)) return false;
        if (nodes_[key].frequency != std::numeric_limits<uint64_t>::max()) ++nodes_[key].frequency;
        return true;
    }

    // Call only AFTER authoritative payload publication succeeds. Validated
    // normal-path operations allocate nothing. Host integration must require
    // max_role_bytes < small_capacity: upstream equality may refuse insertion.
    bool commit_admission(uint32_t key, uint64_t bytes, Admission intent) {
        if (!intent.eligible || key >= nodes_.size() || resident(key) || !bytes || bytes > small_capacity_)
            throw std::logic_error("Invalid S3FIFO admission commit");
        if (resident_bytes() > capacity_-bytes)
            throw std::logic_error("S3FIFO admission exceeds resident byte capacity");
        if (nodes_[key].queue == Queue::ghost) unlink(key);
        if (!intent.ghost_hit && bytes >= small_capacity_) return false; // pinned upstream edge case.
        const Queue destination = intent.ghost_hit || (!has_evicted_ && queues_[1].bytes >= small_capacity_)
            ? Queue::main : Queue::small_queue;
        append(key,destination,bytes,0);
        return true;
    }

    // Select may rotate metadata/promote hot small entries, but NEVER removes
    // the candidate's payload or membership. Prepare required retirement
    // bookkeeping in HostExpertCache BEFORE commit_capacity_eviction().
    uint32_t select_victim() noexcept {
        if (pending_ != none) return pending_;
        if (!resident_bytes()) return none;
        has_evicted_ = true;
        for (;;) {
            if (queues_[2].bytes > main_capacity_ || queues_[1].head == none) {
                while (queues_[2].head != none) {
                    const uint32_t key = queues_[2].head;
                    const auto bytes = nodes_[key].bytes;
                    const auto freq = nodes_[key].frequency;
                    if (!freq) return pending_ = key;
                    unlink(key);
                    append(key,Queue::main,bytes,std::min<uint64_t>(freq,3)-1);
                }
            } else {
                while (queues_[1].head != none) {
                    const uint32_t key = queues_[1].head;
                    if (nodes_[key].frequency < 2) return pending_ = key;
                    const auto bytes = nodes_[key].bytes;
                    unlink(key);
                    append(key,Queue::main,bytes,0); // fresh FIFO insertion resets frequency.
                }
            }
            if (!resident_bytes()) return none;
        }
    }

    void cancel_victim() noexcept { pending_ = none; }
    bool pending_victim_is(uint32_t key) const noexcept {
        return pending_ == key && resident(key);
    }

    // Two-phase commit. This must run AFTER required weak-retirement allocation
    // succeeds and in the same host-cache lock as its authoritative erase.
    void commit_capacity_eviction(uint32_t key) {
        if (pending_ != key || !resident(key)) throw std::logic_error("S3FIFO victim changed before commit");
        const auto queue = nodes_[key].queue;
        const auto bytes = nodes_[key].bytes;
        unlink(key);
        pending_ = none;
        if (queue == Queue::small_queue && bytes <= ghost_capacity_) {
            while (queues_[3].bytes > ghost_capacity_-bytes) unlink(queues_[3].head);
            append(key,Queue::ghost,bytes,0);
        }
    }

    // GPU promotion/take, duplicate replacement, explicit removal: these are
    // NOT capacity misses and do not manufacture a ghost history entry.
    void remove_external(uint32_t key) noexcept {
        if (key >= nodes_.size()) return;
        if (pending_ == key) pending_ = none;
        if (nodes_[key].queue != Queue::absent) unlink(key);
    }

    Snapshot snapshot() const noexcept {
        return {queues_[1].bytes,queues_[2].bytes,queues_[3].bytes,
                queues_[1].count,queues_[2].count,queues_[3].count,has_evicted_};
    }

    uint64_t debug_digest() const noexcept {
        uint64_t hash = 14695981039346656037ull;
        auto add = [&](uint64_t value) {
            for (int i=0;i<8;++i) { hash ^= (value >> (8*i)) & 255; hash *= 1099511628211ull; }
        };
        add(has_evicted_ ? 1 : 0);
        for (unsigned q=1;q<4;++q)
            for (uint32_t key=queues_[q].head;key!=none;key=nodes_[key].next) {
                add(q); add(key); add(nodes_[key].bytes); add(nodes_[key].frequency);
            }
        return hash;
    }

    // Expensive O(key_count) unit/debug audit; never needed in the hot path.
    bool invariant() const noexcept {
        uint32_t active = 0, walked = 0;
        for (const auto & node : nodes_) if (node.queue != Queue::absent) ++active;
        for (unsigned q = 1; q < 4; ++q) {
            uint32_t count = 0, previous = none;
            uint64_t bytes = 0;
            for (uint32_t key = queues_[q].head; key != none; key = nodes_[key].next) {
                if (key >= nodes_.size() || ++count > nodes_.size()) return false;
                const auto & node = nodes_[key];
                if (static_cast<unsigned>(node.queue) != q || node.previous != previous || !node.bytes) return false;
                bytes += node.bytes; previous = key;
            }
            if (count != queues_[q].count || bytes != queues_[q].bytes || previous != queues_[q].tail) return false;
            walked += count;
        }
        return walked == active && resident_bytes() <= capacity_ && queues_[3].bytes <= ghost_capacity_
            && (pending_ == none || resident(pending_));
    }

private:
    static uint32_t checked_key_count(uint32_t keys,uint64_t capacity) {
        if (!keys || keys == none || capacity < 10 || capacity > (uint64_t{1} << 53))
            throw std::invalid_argument("Invalid bounded S3FIFO dimensions/capacity");
        return keys;
    }
    struct Node {
        uint64_t bytes = 0, frequency = 0;
        uint32_t previous = none, next = none;
        Queue queue = Queue::absent;
    };
    struct List { uint32_t head = none, tail = none, count = 0; uint64_t bytes = 0; };

    void unlink(uint32_t key) noexcept {
        auto & node = nodes_[key];
        auto & list = queues_[static_cast<unsigned>(node.queue)];
        if (node.previous == none) list.head = node.next;
        else nodes_[node.previous].next = node.next;
        if (node.next == none) list.tail = node.previous;
        else nodes_[node.next].previous = node.previous;
        list.bytes -= node.bytes; --list.count;
        node = Node{};
    }

    void append(uint32_t key,Queue queue,uint64_t bytes,uint64_t frequency) noexcept {
        auto & list = queues_[static_cast<unsigned>(queue)];
        auto & node = nodes_[key];
        node = Node{bytes,frequency,list.tail,none,queue};
        if (list.tail == none) list.head = key;
        else nodes_[list.tail].next = key;
        list.tail = key; ++list.count; list.bytes += bytes;
    }

    std::vector<Node> nodes_; // The ONLY allocation; fixed after construction.
    uint64_t capacity_,small_capacity_,main_capacity_,ghost_capacity_;
    List queues_[4];
    uint32_t pending_ = none;
    bool has_evicted_ = false;
};

} // namespace questwend
