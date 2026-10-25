#pragma once
#include "gpu_lru_index.h"
#include <stdexcept>

namespace questwend {

// Shared by actual ExpertCache mutations and model-free transaction tests.
// Pool payloads remain authoritative; the optional index contains metadata only.
// Both caller paths validate expected occupied/free state BEFORE mutation.
template<class Pool>
void commit_gpu_lru_clock(Pool & pool, int slot, uint64_t & clock) {
    const uint64_t next = clock + 1; // Preserve the original unsigned wrap behavior.
    if (pool.lru_index) pool.lru_index->occupy_or_touch(slot, next);
    pool.clk[slot] = next;
    clock = next;
}

template<class Pool>
void record_gpu_lru_access(Pool & pool, int slot, uint64_t & clock) {
    if (slot < 0 || slot >= pool.n_slots || pool.slot2key[slot] < 0 ||
        (pool.lru_index && (!pool.lru_index->valid_retained_slot(slot) || !pool.lru_index->is_occupied(slot))))
        throw std::logic_error("GPU LRU access outside an occupied retained slot");
    commit_gpu_lru_clock(pool, slot, clock);
}

template<class Pool>
void publish_gpu_lru_slot(Pool & pool, int slot, int key, int layer, uint64_t & clock) {
    if (slot < 0 || slot >= pool.n_slots || pool.slot2key[slot] >= 0 ||
        key < 0 || static_cast<size_t>(key) >= pool.slot_of.size() ||
        layer < 0 || static_cast<size_t>(layer) >= pool.occ.size() ||
        (pool.lru_index && (!pool.lru_index->valid_retained_slot(slot) || pool.lru_index->is_occupied(slot))))
        throw std::logic_error("Invalid GPU LRU slot publication");
    // This is the one potentially allocating map operation that existed in the
    // original publication path. Do it BEFORE publishing slot/occupancy/remap
    // state. If it throws, the already released victim stays a valid free slot.
    const auto inserted = pool.key2slot.emplace(key, slot);
    if (!inserted.second) throw std::logic_error("GPU LRU key was already published");
    // All dimensions were checked; index mutations and scalar writes allocate
    // nothing. There is no new fallible operation after this map commit.
    pool.slot2key[slot] = key;
    ++pool.occ[layer];
    pool.slot_of[key] = slot;
    commit_gpu_lru_clock(pool, slot, clock);
}

template<class Pool>
int validate_gpu_lru_drop(const Pool & pool, int slot, int experts_per_layer) {
    if (slot < 0 || slot >= pool.n_slots || experts_per_layer <= 0 ||
        (pool.lru_index && !pool.lru_index->valid_retained_slot(slot)))
        throw std::logic_error("Invalid GPU LRU slot removal");
    const int key = pool.slot2key[slot];
    if (key < 0) {
        if (pool.lru_index && pool.lru_index->is_occupied(slot))
            throw std::logic_error("GPU LRU free payload slot has an occupied index entry");
        return -1;
    }
    const int layer = key / experts_per_layer;
    const auto found = pool.key2slot.find(key);
    if (static_cast<size_t>(key) >= pool.slot_of.size() ||
        static_cast<size_t>(layer) >= pool.occ.size() || pool.occ[layer] <= 0 ||
        found == pool.key2slot.end() || found->second != slot ||
        (pool.lru_index && !pool.lru_index->is_occupied(slot)))
        throw std::logic_error("GPU LRU slot maps disagree before removal");
    return key;
}

template<class Pool>
int drop_gpu_lru_slot(Pool & pool, int slot, int experts_per_layer) {
    const int key = validate_gpu_lru_drop(pool, slot, experts_per_layer);
    if (key < 0) return -1;
    const int layer = key / experts_per_layer;
    const auto found = pool.key2slot.find(key);
    if (pool.lru_index) pool.lru_index->release(slot);
    --pool.occ[layer];
    pool.slot_of[key] = 0;
    pool.key2slot.erase(found);
    pool.slot2key[slot] = -1;
    return key;
}

} // namespace questwend
