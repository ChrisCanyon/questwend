#pragma once

#include <cstdint>
#include <vector>

namespace questwend {

// One immutable, sampled layer view. Each bit corresponds to gate/up/down.
// A role with neither bit is backed by the model file on SSD. These masks
// describe storage placement, not whether a transient GPU scratch copy is
// currently being used for a matrix multiplication.
struct ExpertTelemetry {
    int layer = -1;
    int layers = 0;
    int experts = 0;
    uint64_t sampled_at_unix_ms = 0;
    uint64_t sequence = 0;
    bool observations_available = false;
    uint64_t ram_capacity_bytes = 0;
    uint64_t ram_resident_bytes = 0;
    uint64_t capture_microseconds = 0;
    uint64_t capture_total_microseconds = 0;
    uint64_t capture_samples = 0;
    std::vector<uint8_t> gpu_mask;
    std::vector<uint8_t> ram_mask;
    std::vector<uint64_t> observations;
};

} // namespace questwend
