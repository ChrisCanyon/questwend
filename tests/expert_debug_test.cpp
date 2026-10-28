#include "expert_debug.h"

#include <cassert>
#include <cstdint>
#include <thread>
#include <chrono>

using questwend::ExpertDebug;

int main() {
    auto & debug = ExpertDebug::instance();
    debug.initialize(2, 4);
    const int32_t selected[] = {1, 2, 2};

    assert(debug.mode() == ExpertDebug::Off);
    debug.observe(1, selected, 3);
    assert(debug.snapshot().observations[5] == 0);

    debug.configure(ExpertDebug::Overview, 1000);
    debug.set_phase(ExpertDebug::Prefill);
    assert(debug.snapshot().phase == "prefill");
    debug.set_phase(ExpertDebug::Decode);
    assert(debug.snapshot().phase == "decode");
    debug.set_phase(ExpertDebug::Idle);
    assert(debug.snapshot().phase == "idle");
    debug.observe(1, selected, 3);
    assert(debug.snapshot().observations[5] == 1);
    assert(debug.snapshot().observations[6] == 2);
    uint64_t dropped = 0;
    assert(debug.events_since(0, dropped).empty());

    debug.configure(ExpertDebug::Detailed, 1000);
    debug.layer_stage(1, "batch_layer", 100);
    debug.observe(1, selected, 3);
    const auto events = debug.events_since(0, dropped);
    assert(events.size() == 3);
    assert(events[0].kind == "phase" && events[0].count == ExpertDebug::Idle);
    assert(events[1].kind == "batch_layer");
    assert(events[2].kind == "selection");
    assert(events[2].count == 3);

    debug.set_phase(ExpertDebug::Decode);
    const auto phase = debug.events_since(events.back().sequence, dropped);
    assert(phase.size() == 1 && phase[0].kind == "phase");
    assert(phase[0].count == ExpertDebug::Decode);

    debug.scratch(1, {2, 7, 3});
    debug.execution(1, "gpu", true, selected, 3);
    debug.execution(1, "cpu", true, selected, 1);
    debug.execution(1, "cpu", false, selected, 1);
    debug.execution(1, "gpu", false, selected, 3);
    const auto detailed = debug.events_since(phase.back().sequence, dropped);
    assert(detailed.size() == 5);
    assert(detailed[0].kind == "scratch" && detailed[0].values == std::vector<int>({2, 7, 3}));
    assert(detailed[1].kind == "gpu_begin" && detailed[1].values.size() == 3);
    assert(detailed[2].kind == "cpu_begin" && detailed[3].kind == "cpu_end");
    assert(detailed[4].kind == "gpu_end");

    debug.configure(ExpertDebug::Off, 0);
    assert(debug.mode() == ExpertDebug::Off);
    assert(debug.events_since(0, dropped).empty());
    assert(debug.snapshot().observations[5] == 0);

    debug.configure(ExpertDebug::Overview, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    assert(debug.mode() == ExpertDebug::Off);
}
