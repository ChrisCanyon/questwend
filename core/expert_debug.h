#pragma once

// Optional, process-local visualization recorder. No file or network I/O occurs
// in inference hooks. The normal path checks one atomic and returns.

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace questwend {

class ExpertCache;

struct ExpertDebugEvent {
    uint64_t sequence = 0;
    uint64_t at_ms = 0;
    std::string kind;
    int layer = -1;
    int count = 0;
    // Selection: {expert, hits}; placement: {layer, expert, role, from, to};
    // scratch: {expert, role mask, RAM source mask}; execution: expert IDs.
    std::vector<int> values;
};

struct ExpertDebugSnapshot {
    int layers = 0;
    int experts = 0;
    uint64_t sequence = 0;
    uint64_t sampled_at_ms = 0;
    uint64_t capture_us = 0;
    uint64_t capture_total_us = 0;
    uint64_t capture_samples = 0;
    uint64_t dropped_events = 0;
    std::string phase = "idle";
    std::vector<uint8_t> gpu;
    std::vector<uint8_t> ram;
    std::vector<uint32_t> observations;
};

class ExpertDebug {
public:
    enum Mode { Off = 0, Overview = 1, Detailed = 2 };
    enum Phase { Idle = 0, Prefill = 1, Decode = 2 };
    static ExpertDebug & instance();

    void configure(Mode mode, uint64_t ttl_ms);
    void set_phase(Phase phase);
    Mode mode() const;
    void initialize(int layers, int experts);
    void observe(int layer, const int32_t * ids, int count);
    void layer_stage(int layer, const char * stage, int tokens);
    void scratch(int layer, const std::vector<int> & parts);
    void execution(int layer, const char * device, bool begin,
                   const int32_t * experts, int count);
    void maybe_capture(const ExpertCache & cache, bool force = false);
    ExpertDebugSnapshot snapshot() const;
    std::vector<ExpertDebugEvent> events_since(uint64_t sequence, uint64_t & dropped) const;

private:
    static uint64_t now_ms();
    void push_event(ExpertDebugEvent event);

    std::atomic<int> mode_{Off};
    std::atomic<int> phase_{Idle};
    std::atomic<uint64_t> until_ms_{0};
    std::atomic<uint64_t> next_capture_ms_{0};
    mutable std::mutex mutex_;
    ExpertDebugSnapshot snapshot_;
    std::deque<ExpertDebugEvent> events_;
    uint64_t next_event_ = 0;
    static constexpr size_t max_events_ = 32768;
};

} // namespace questwend
