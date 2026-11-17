#pragma once

// Metadata-only accelerator for the existing unconstrained GPU LRU scan.
// This changes neither admission policy nor the exact (clock, slot) victim.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace questwend {

class GpuLruIndex {
public:
    struct Selection {
        int slot = -1;
        uint64_t occupied_examined = 0;
        uint64_t bitmap_words_examined = 0;
    };

    // Allocate for the physical pool once. A workspace repartition only changes
    // its retained prefix, so rebuild needs no new storage or payload copies.
    explicit GpuLruIndex(size_t maximum_slots)
        : nodes_(checked_slots(maximum_slots)), free_words_((nodes_.size() + 63) / 64),
          free_summary_((free_words_.size() + 63) / 64), sort_scratch_(nodes_.size()) {}

    template<class Keys, class Clocks>
    void rebuild(int retained_slots, const Keys & keys, const Clocks & clocks) {
        if (retained_slots < 0 || static_cast<size_t>(retained_slots) > nodes_.size() ||
            keys.size() < static_cast<size_t>(retained_slots) || clocks.size() < static_cast<size_t>(retained_slots))
            throw std::invalid_argument("GPU LRU index rebuild dimensions do not match the retained pool");
        retained_slots_ = retained_slots;
        head_ = tail_ = -1;
        occupied_ = 0;
        std::fill(free_words_.begin(), free_words_.end(), uint64_t{0});
        std::fill(free_summary_.begin(), free_summary_.end(), uint64_t{0});
        for (int slot = 0; slot < retained_slots_; ++slot) {
            auto & node = nodes_[slot];
            node = Node{};
            node.clock = clocks[slot];
            if (keys[slot] < 0) set_free(slot, true);
            else {
                node.occupied = true;
                sort_scratch_[occupied_++] = slot;
            }
        }
        std::sort(sort_scratch_.begin(), sort_scratch_.begin() + occupied_,
            [this](int left, int right) { return older(left, right); });
        for (int index = 0; index < occupied_; ++index) append(sort_scratch_[index]);
    }

    void occupy_or_touch(int slot, uint64_t clock) {
        require_slot(slot);
        auto & node = nodes_[slot];
        if (node.occupied) unlink(slot);
        else {
            node.occupied = true;
            ++occupied_;
            set_free(slot, false);
        }
        node.clock = clock;
        // Normally ++clock_ is globally newer: constant-time append. The rare
        // ordered insertion also preserves scan ties and uint64 clock wrap.
        if (tail_ < 0 || older(tail_, slot)) append(slot);
        else {
            int after = head_;
            while (after >= 0 && older(after, slot)) after = nodes_[after].next;
            if (after < 0) append(slot);
            else insert_before(slot, after);
        }
    }

    void release(int slot) {
        require_slot(slot);
        auto & node = nodes_[slot];
        if (!node.occupied) return;
        unlink(slot);
        node.occupied = false;
        --occupied_;
        set_free(slot, true);
    }

    bool valid_retained_slot(int slot) const noexcept { return slot >= 0 && slot < retained_slots_; }
    bool is_occupied(int slot) const noexcept {
        return valid_retained_slot(slot) && nodes_[slot].occupied;
    }

    // Must not mutate ordering: can_retain() peeks repeatedly, and protected
    // selected experts remain protected until the existing ActiveExpertGuard
    // ends. is_protected receives a SLOT; the owner reads slot2key[slot].
    template<class IsProtected>
    Selection pick(IsProtected is_protected) const {
        Selection result;
        for (size_t summary = 0; summary < free_summary_.size(); ++summary) {
            ++result.bitmap_words_examined;
            if (!free_summary_[summary]) continue;
            const size_t word = summary * 64 + first_set_bit(free_summary_[summary]);
            ++result.bitmap_words_examined;
            result.slot = static_cast<int>(word * 64 + first_set_bit(free_words_[word]));
            return result;
        }
        for (int slot = head_; slot >= 0; slot = nodes_[slot].next) {
            ++result.occupied_examined;
            // The original scan initializes best=UINT64_MAX and uses strict <.
            // An occupied max-clock slot therefore never wins that scan.
            if (nodes_[slot].clock == std::numeric_limits<uint64_t>::max()) return result;
            if (!is_protected(slot)) { result.slot = slot; return result; }
        }
        return result;
    }

    size_t metadata_allocation_bytes() const noexcept {
        return nodes_.capacity() * sizeof(Node) + sort_scratch_.capacity() * sizeof(int) +
            (free_words_.capacity() + free_summary_.capacity()) * sizeof(uint64_t);
    }

    // Unit/profile-boundary check only; no need for a full scan in pick().
    template<class Keys, class Clocks>
    bool invariant(const Keys & keys, const Clocks & clocks) const noexcept {
        if (keys.size() < static_cast<size_t>(retained_slots_) || clocks.size() < static_cast<size_t>(retained_slots_))
            return false;
        int linked = 0, previous = -1;
        for (int slot = head_; slot >= 0; slot = nodes_[slot].next) {
            if (slot >= retained_slots_ || ++linked > occupied_) return false;
            const auto & node = nodes_[slot];
            if (!node.occupied || node.previous != previous || keys[slot] < 0 || node.clock != clocks[slot]) return false;
            if (previous >= 0 && !older(previous, slot)) return false;
            previous = slot;
        }
        if (linked != occupied_ || previous != tail_) return false;
        int occupied_keys = 0;
        for (int slot = 0; slot < retained_slots_; ++slot) {
            const bool free = (free_words_[slot / 64] >> (slot % 64)) & 1;
            if (nodes_[slot].occupied == free || free != (keys[slot] < 0)) return false;
            occupied_keys += keys[slot] >= 0;
        }
        if (occupied_keys != occupied_) return false;
        for (size_t word = 0; word < free_words_.size(); ++word)
            if (((free_summary_[word / 64] >> (word % 64)) & 1) != (free_words_[word] != 0)) return false;
        return true;
    }

private:
    static size_t checked_slots(size_t maximum_slots) {
        if (maximum_slots > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("GPU LRU index has too many slots");
        return maximum_slots;
    }

    struct Node {
        uint64_t clock = 0;
        int previous = -1, next = -1;
        bool occupied = false;
    };

    static unsigned first_set_bit(uint64_t bits) noexcept {
        // Portable bounded fallback; can use the platform bit-scan intrinsic in
        // a later independent micro-optimization. Callers guarantee bits != 0.
        unsigned index = 0;
        while (!(bits & 1)) { bits >>= 1; ++index; }
        return index;
    }

    void require_slot(int slot) const {
        if (slot < 0 || slot >= retained_slots_)
            throw std::out_of_range("GPU LRU index slot is outside the retained prefix");
    }

    bool older(int left, int right) const noexcept {
        return nodes_[left].clock < nodes_[right].clock ||
            (nodes_[left].clock == nodes_[right].clock && left < right);
    }

    void set_free(int slot, bool free) noexcept {
        const size_t word = static_cast<size_t>(slot) / 64;
        const uint64_t bit = uint64_t{1} << (slot % 64);
        if (free) free_words_[word] |= bit;
        else free_words_[word] &= ~bit;
        const uint64_t summary_bit = uint64_t{1} << (word % 64);
        if (free_words_[word]) free_summary_[word / 64] |= summary_bit;
        else free_summary_[word / 64] &= ~summary_bit;
    }

    void unlink(int slot) noexcept {
        auto & node = nodes_[slot];
        if (node.previous < 0) head_ = node.next;
        else nodes_[node.previous].next = node.next;
        if (node.next < 0) tail_ = node.previous;
        else nodes_[node.next].previous = node.previous;
        node.previous = node.next = -1;
    }

    void append(int slot) noexcept {
        nodes_[slot].previous = tail_;
        nodes_[slot].next = -1;
        if (tail_ < 0) head_ = slot;
        else nodes_[tail_].next = slot;
        tail_ = slot;
    }

    void insert_before(int slot, int after) noexcept {
        auto & node = nodes_[slot];
        node.previous = nodes_[after].previous;
        node.next = after;
        if (node.previous < 0) head_ = slot;
        else nodes_[node.previous].next = slot;
        nodes_[after].previous = slot;
    }

    std::vector<Node> nodes_;
    std::vector<uint64_t> free_words_, free_summary_;
    std::vector<int> sort_scratch_;
    int retained_slots_ = 0, head_ = -1, tail_ = -1, occupied_ = 0;
};

} // namespace questwend
