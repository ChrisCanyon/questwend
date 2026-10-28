#include "expert_debug.h"
#include "expert_cache.h"

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace questwend {

namespace {
uint64_t wall_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
}

ExpertDebug & ExpertDebug::instance() {
    static ExpertDebug debug;
    return debug;
}

uint64_t ExpertDebug::now_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void ExpertDebug::configure(Mode requested, uint64_t ttl_ms) {
    const Mode previous = mode();
    const bool previously_active = previous != Off;
    if (requested < Off || requested > Detailed) requested = Off;
    ttl_ms = std::min<uint64_t>(ttl_ms, 60000);
    if (requested != Off && ttl_ms == 0) requested = Off;
    until_ms_.store(requested == Off ? 0 : now_ms() + ttl_ms, std::memory_order_release);
    mode_.store(requested, std::memory_order_release);
    if (requested == Off || !previously_active) {
        std::lock_guard<std::mutex> guard(mutex_);
        events_.clear();
        snapshot_.observations.assign(snapshot_.observations.size(), 0);
        snapshot_.sequence = 0;
        snapshot_.sampled_at_ms = 0;
        next_capture_ms_.store(0, std::memory_order_relaxed);
    }
    if (requested == Detailed && previous != Detailed) {
        std::lock_guard<std::mutex> guard(mutex_);
        ExpertDebugEvent event;
        event.kind = "phase";
        event.count = phase_.load(std::memory_order_relaxed);
        push_event(std::move(event));
    }
}

void ExpertDebug::set_phase(Phase phase) {
    const int previous = phase_.exchange(phase, std::memory_order_relaxed);
    if (previous == phase || mode() != Detailed) return;
    std::lock_guard<std::mutex> guard(mutex_);
    ExpertDebugEvent event;
    event.kind = "phase";
    event.count = phase;
    push_event(std::move(event));
}

ExpertDebug::Mode ExpertDebug::mode() const {
    const auto current = static_cast<Mode>(mode_.load(std::memory_order_acquire));
    if (current == Off || now_ms() >= until_ms_.load(std::memory_order_acquire)) return Off;
    return current;
}

void ExpertDebug::initialize(int layers, int experts) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (snapshot_.layers == layers && snapshot_.experts == experts) return;
    snapshot_ = {};
    snapshot_.layers = layers;
    snapshot_.experts = experts;
    const size_t count = static_cast<size_t>(layers) * experts;
    snapshot_.gpu.assign(count, 0);
    snapshot_.ram.assign(count, 0);
    snapshot_.observations.assign(count, 0);
    events_.clear();
    next_event_ = 0;
}

void ExpertDebug::push_event(ExpertDebugEvent event) {
    event.sequence = ++next_event_;
    event.at_ms = wall_ms();
    if (events_.size() == max_events_) {
        events_.pop_front();
        ++snapshot_.dropped_events;
    }
    events_.push_back(std::move(event));
}

void ExpertDebug::observe(int layer, const int32_t * ids, int count) {
    const auto current = mode();
    if (current == Off || count <= 0) return;
    std::lock_guard<std::mutex> guard(mutex_);
    if (layer < 0 || layer >= snapshot_.layers) return;
    std::unordered_map<int, int> histogram;
    const size_t offset = static_cast<size_t>(layer) * snapshot_.experts;
    for (int index = 0; index < count; ++index) {
        const int expert = ids[index];
        if (expert < 0 || expert >= snapshot_.experts) continue;
        ++snapshot_.observations[offset + expert];
        if (current == Detailed) ++histogram[expert];
    }
    if (current == Detailed && !histogram.empty()) {
        ExpertDebugEvent event;
        event.kind = "selection";
        event.layer = layer;
        event.count = count;
        event.values.reserve(histogram.size() * 2);
        for (const auto & [expert, hits] : histogram) {
            event.values.push_back(expert);
            event.values.push_back(hits);
        }
        push_event(std::move(event));
    }
}

void ExpertDebug::layer_stage(int layer, const char * stage, int tokens) {
    if (mode() != Detailed) return;
    std::lock_guard<std::mutex> guard(mutex_);
    ExpertDebugEvent event;
    event.kind = stage;
    event.layer = layer;
    event.count = tokens;
    push_event(std::move(event));
}

void ExpertDebug::scratch(int layer, const std::vector<int> & parts) {
    if (mode() != Detailed || parts.empty()) return;
    std::lock_guard<std::mutex> guard(mutex_);
    ExpertDebugEvent event;
    event.kind = "scratch";
    event.layer = layer;
    event.count = static_cast<int>(parts.size() / 3);
    event.values = parts;
    push_event(std::move(event));
}

void ExpertDebug::execution(int layer, const char * device, bool begin,
                            const int32_t * experts, int count) {
    if (mode() != Detailed || count <= 0) return;
    std::lock_guard<std::mutex> guard(mutex_);
    ExpertDebugEvent event;
    event.kind = std::string(device) + (begin ? "_begin" : "_end");
    event.layer = layer;
    event.count = count;
    event.values.assign(experts, experts + count);
    push_event(std::move(event));
}

void ExpertDebug::maybe_capture(const ExpertCache & cache, bool force) {
    const auto current = mode();
    if (current == Off) return;
    const uint64_t now = now_ms();
    if (!force) {
        uint64_t next = next_capture_ms_.load(std::memory_order_relaxed);
        if (now < next || !next_capture_ms_.compare_exchange_strong(
                next, now + (current == Detailed ? 50 : 500), std::memory_order_relaxed)) return;
    }
    const auto started = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> guard(mutex_);
    const bool had_sample = snapshot_.sequence != 0;
    const auto old_gpu = current == Detailed && had_sample ? snapshot_.gpu : std::vector<uint8_t>{};
    const auto old_ram = current == Detailed && had_sample ? snapshot_.ram : std::vector<uint8_t>{};
    cache.debug_placement(snapshot_.gpu, snapshot_.ram);
    if (current == Detailed && !old_gpu.empty()) {
        ExpertDebugEvent moves;
        moves.kind = "placement";
        for (size_t index = 0; index < snapshot_.gpu.size(); ++index) {
            for (int role = 0; role < 3; ++role) {
                const int bit = 1 << role;
                const int before = old_gpu[index] & bit ? 0 : old_ram[index] & bit ? 1 : 2;
                const int after = snapshot_.gpu[index] & bit ? 0 : snapshot_.ram[index] & bit ? 1 : 2;
                if (before == after) continue;
                moves.values.insert(moves.values.end(), {
                    static_cast<int>(index / snapshot_.experts),
                    static_cast<int>(index % snapshot_.experts), role, before, after});
            }
        }
        moves.count = static_cast<int>(moves.values.size() / 5);
        if (moves.count) push_event(std::move(moves));
    }
    snapshot_.sampled_at_ms = wall_ms();
    ++snapshot_.sequence;
    snapshot_.capture_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count());
    snapshot_.capture_total_us += snapshot_.capture_us;
    ++snapshot_.capture_samples;
}

ExpertDebugSnapshot ExpertDebug::snapshot() const {
    std::lock_guard<std::mutex> guard(mutex_);
    auto result = snapshot_;
    switch (phase_.load(std::memory_order_relaxed)) {
        case Prefill: result.phase = "prefill"; break;
        case Decode: result.phase = "decode"; break;
        default: result.phase = "idle"; break;
    }
    return result;
}

std::vector<ExpertDebugEvent> ExpertDebug::events_since(uint64_t sequence, uint64_t & dropped) const {
    std::lock_guard<std::mutex> guard(mutex_);
    dropped = snapshot_.dropped_events;
    std::vector<ExpertDebugEvent> result;
    for (const auto & event : events_) {
        if (event.sequence > sequence) result.push_back(event);
        if (result.size() >= 256) break;
    }
    return result;
}

} // namespace questwend
