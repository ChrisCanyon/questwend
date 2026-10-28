#include "expert_cache.h"
#include "expert_debug.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <utility>

namespace questwend {

void ExpertCache::request_telemetry_layer(int layer) {
    telemetry_requested_layer_.store(layer, std::memory_order_relaxed);
}

ExpertTelemetry ExpertCache::telemetry_snapshot() const {
    std::lock_guard<std::mutex> guard(telemetry_mutex_);
    return telemetry_;
}

void ExpertCache::debug_placement(std::vector<uint8_t> & gpu,
                                  std::vector<uint8_t> & ram) const {
    const size_t size = static_cast<size_t>(n_layer_) * n_expert_;
    if (gpu.size() != size || ram.size() != size) return;
    for (int layer = 0; layer < n_layer_; ++layer) {
        if (!serves(layer)) continue;
        const size_t start = static_cast<size_t>(layer) * n_expert_;
        std::fill_n(gpu.begin() + start, n_expert_, 0);
        std::fill_n(ram.begin() + start, n_expert_, 0);
        for (int role = 0; role < N_ROLE; ++role) {
            const auto present = host_cache_
                ? host_cache_->presence_range(ram_key(role, layer, 0), n_expert_)
                : std::vector<uint8_t>{};
            for (int expert = 0; expert < n_expert_; ++expert) {
                const size_t key = start + expert;
                if (resident(static_cast<Role>(role), layer, expert))
                    gpu[key] |= static_cast<uint8_t>(1 << role);
                else if (!present.empty() && present[expert])
                    ram[key] |= static_cast<uint8_t>(1 << role);
            }
        }
    }
}

void ExpertCache::refresh_telemetry(bool force) {
    const int selected = telemetry_requested_layer_.load(std::memory_order_relaxed);
    if (!serves(selected)) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && telemetry_last_ != std::chrono::steady_clock::time_point{} &&
        now - telemetry_last_ < std::chrono::milliseconds(500)) return;
    telemetry_last_ = now;

    ExpertTelemetry next;
    next.layer = selected;
    next.layers = n_layer_;
    next.experts = n_expert_;
    next.sampled_at_unix_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    next.observations_available = activation_ != nullptr;
    next.gpu_mask.assign(n_expert_, 0);
    next.ram_mask.assign(n_expert_, 0);
    next.observations.assign(n_expert_, 0);

    for (int role = 0; role < N_ROLE; ++role) {
        const auto present = host_cache_
            ? host_cache_->presence_range(ram_key(role, selected, 0), n_expert_)
            : std::vector<uint8_t>{};
        for (int expert = 0; expert < n_expert_; ++expert) {
            if (resident(static_cast<Role>(role), selected, expert))
                next.gpu_mask[expert] |= static_cast<uint8_t>(1 << role);
            if (!present.empty() && present[expert])
                next.ram_mask[expert] |= static_cast<uint8_t>(1 << role);
        }
    }
    if (activation_)
        for (int expert = 0; expert < n_expert_; ++expert)
            next.observations[expert] = activation_->observed_count(selected, expert);
    if (host_cache_) {
        const auto usage = host_cache_->stats();
        next.ram_capacity_bytes = usage.capacity_bytes;
        next.ram_resident_bytes = usage.resident_bytes;
    }

    next.capture_microseconds = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - now).count());
    telemetry_total_microseconds_ += next.capture_microseconds;
    next.capture_total_microseconds = telemetry_total_microseconds_;
    next.capture_samples = ++telemetry_samples_;

    std::lock_guard<std::mutex> guard(telemetry_mutex_);
    next.sequence = telemetry_.sequence + 1;
    telemetry_ = std::move(next);
}

} // namespace questwend
