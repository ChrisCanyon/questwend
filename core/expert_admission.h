// TinyLFU admission adapted from Caffeine, Apache License 2.0.
// Caffeine commit e8fc857ef523bd04439661d0265f28bbe29f0848:
// BoundedLocalCache.admit and FrequencySketch. Retain third_party/caffeine/LICENSE.
// Changes: exact dense expert IDs instead of a hash sketch; wider counters;
// every logical demand advances aging, including CPU/transient execution.
// This is the admission rule, not the complete Window TinyLFU cache policy.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace questwend {

class ExpertAdmission {
public:
    ExpertAdmission(size_t expert_count, size_t gpu_capacity)
        : frequencies_(expert_count, 0), window_(10 * std::max<size_t>(256, gpu_capacity)) {}

    // Keep observed frequencies when retained capacity changes. Age to the
    // new sample horizon instead of resetting hot-expert history.
    void set_capacity(size_t gpu_capacity) {
        window_ = 10 * std::max<size_t>(256, gpu_capacity);
        while (samples_ >= window_) {
            for (auto & frequency : frequencies_) frequency >>= 1;
            samples_ >>= 1;
        }
    }

    void observe(size_t expert) {
        if (expert >= frequencies_.size()) throw std::out_of_range("Invalid expert demand ID");
        if (frequencies_[expert] < std::numeric_limits<uint32_t>::max()) ++frequencies_[expert];
        if (++samples_ >= window_) {
            for (auto & frequency : frequencies_) frequency >>= 1;
            samples_ >>= 1;
        }
    }

    uint32_t frequency(size_t expert) const { return frequencies_.at(expert); }

    bool admit(size_t candidate, size_t victim) const {
        return frequency(candidate) > frequency(victim);
    }

private:
    std::vector<uint32_t> frequencies_;
    uint64_t samples_ = 0, window_;
};

} // namespace questwend
