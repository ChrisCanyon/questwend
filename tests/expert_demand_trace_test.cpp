#include "expert_demand_trace.h"
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <new>
#include <stdexcept>
#include <vector>

static std::atomic<bool> deny_allocations{false};
static std::atomic<unsigned> denied{0};

void * operator new(std::size_t bytes) {
    if (deny_allocations.load()) { ++denied; throw std::bad_alloc(); }
    if (void * memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void * operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void * memory) noexcept { std::free(memory); }
void operator delete[](void * memory) noexcept { std::free(memory); }
void operator delete(void * memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void * memory, std::size_t) noexcept { std::free(memory); }

using questwend::ExpertDemandTrace;
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
static uint16_t u16(const uint8_t * p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
static uint32_t u32(const uint8_t * p) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= uint32_t(p[i]) << (8 * i);
    return value;
}
static uint64_t u64(const uint8_t * p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}

static void no_hot_allocations_and_position_rewinds() {
    ExpertDemandTrace trace(2, 512, 2, {80, 80, 160, 80, 80, 320}, 4096);
    int32_t ids[2]{511, 7};
    deny_allocations.store(true);
    {
        auto call = trace.begin(32000, 1, ExpertDemandTrace::Phase::single_forward);
        call.layer(0, ids, 1, 2);
        call.layer(1, ids, 1, 2);
    }
    {
        auto call = trace.begin(31, 1, ExpertDemandTrace::Phase::single_forward);
        call.layer(0, ids, 1, 2);
        call.layer(1, ids, 1, 2);
    }
    deny_allocations.store(false);
    require(denied == 0 && !trace.invalid_trace(), "Trace hot path allocated or failed");
    const auto * bytes = trace.records();
    const size_t stride = 32 + 2 * 2 * 2;
    require(trace.used_bytes() == 2 * stride, "Wrong fixed frame length");
    require(u64(bytes) == 0 && u64(bytes + stride) == 1, "Call IDs rewound with context position");
    require(u32(bytes + 8) == 32000 && u32(bytes + stride + 8) == 31, "Source positions lost");
    require(u16(bytes + 22) == ExpertDemandTrace::complete && u16(bytes + 32) == 511,
            "Completion or uint16 expert ID framing changed");
    require(ids[0] == 511 && ids[1] == 7, "Trace mutated router input");
}

static void markers_and_full_layer_major_batches() {
    ExpertDemandTrace markers(2, 8, 2, {10, 10, 20, 10, 10, 40}, 4096);
    { auto call = markers.begin(0, 4096, ExpertDemandTrace::Phase::prefill_batch); }
    require(markers.used_bytes() == 32 && u32(markers.records() + 24) == 0,
            "Default prefill capture consumed unbounded routing payload");
    require(u16(markers.records() + 22) == (ExpertDemandTrace::complete | ExpertDemandTrace::marker_only),
            "Prefill marker flags missing");
    ExpertDemandTrace full(2, 8, 2, {10, 10, 20, 10, 10, 40}, 4096, true);
    int32_t first[6]{0, 1, 2, 3, 4, 5}, second[6]{7, 6, 5, 4, 3, 2};
    {
        auto call = full.begin(100, 3, ExpertDemandTrace::Phase::prefill_batch);
        call.layer(0, first, 3, 2);
        call.layer(1, second, 3, 2);
    }
    require(u32(full.records() + 24) == 24, "Batch payload length wrong");
    for (int i = 0; i < 6; ++i) {
        require(u16(full.records() + 32 + i * 2) == first[i], "Layer0 row order lost");
        require(u16(full.records() + 44 + i * 2) == second[i], "Layer1 row order lost");
    }
}

static void overflow_and_malformed_calls_are_explicit() {
    ExpertDemandTrace trace(1, 8, 2, {10, 10, 20}, 36);
    int32_t ids[2]{3, 5};
    { auto call = trace.begin(0, 1, ExpertDemandTrace::Phase::single_forward); call.layer(0, ids, 1, 2); }
    const size_t used = trace.used_bytes();
    { auto call = trace.begin(1, 1, ExpertDemandTrace::Phase::single_forward); call.layer(0, ids, 1, 2); }
    require(trace.truncated() && trace.dropped_calls() == 1 && trace.used_bytes() == used,
            "Overflow was not bounded/explicit");
    ExpertDemandTrace prefix(1, 8, 2, {10, 10, 20}, 100, true);
    { auto call = prefix.begin(0, 1, ExpertDemandTrace::Phase::single_forward); call.layer(0, ids, 1, 2); }
    { auto call = prefix.begin(1, 32, ExpertDemandTrace::Phase::prefill_batch); }
    { auto call = prefix.begin(33, 1, ExpertDemandTrace::Phase::single_forward); call.layer(0, ids, 1, 2); }
    require(prefix.used_bytes() == 36 && prefix.dropped_calls() == 2,
            "Trace resumed after an overflow and created a chronological hole");
    ExpertDemandTrace invalid(1, 8, 2, {10, 10, 20}, 128);
    int32_t bad[2]{0, 8};
    { auto call = invalid.begin(0, 1, ExpertDemandTrace::Phase::single_forward); call.layer(0, bad, 1, 2); }
    require(invalid.invalid_trace() && !(u16(invalid.records() + 22) & ExpertDemandTrace::complete),
            "Invalid ID was silently treated as a complete trace");
    ExpertDemandTrace missing(2, 8, 2, {10, 10, 20, 10, 10, 20}, 128);
    { auto call = missing.begin(0, 1, ExpertDemandTrace::Phase::single_forward); call.layer(0, ids, 1, 2); }
    require(missing.invalid_trace(), "Missing logical layer was not detected");
    ExpertDemandTrace aborted(1, 8, 2, {10, 10, 20}, 128);
    try {
        auto call = aborted.begin(0, 1, ExpertDemandTrace::Phase::verify_batch);
        call.layer(0, ids, 1, 2);
        throw std::runtime_error("Injected forward failure");
    } catch (const std::runtime_error &) {}
    require((u16(aborted.records() + 22) & ExpertDemandTrace::aborted) &&
            !(u16(aborted.records() + 22) & ExpertDemandTrace::complete),
            "An unwinding forward was silently marked complete");
}

static void export_header_and_variable_weight_sizes(const char * path) {
    ExpertDemandTrace trace(2, 8, 2, {10, 11, 12, 20, 21, 22}, 128);
    { auto call = trace.begin(10, 3, ExpertDemandTrace::Phase::prefill_batch); }
    require(trace.save(path), "Explicit trace export failed");
    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>{});
    require(bytes.size() == 64 + 8 + 6 * 8 + 32, "Export length/header framing wrong");
    require(bytes[0] == 'Q' && bytes[7] == '1' && u32(bytes.data() + 12) == 0x01020304,
            "Magic/endian marker mismatch");
    require(u64(bytes.data() + 24) == 32 && u64(bytes.data() + 64) == 3,
            "Record length/MoE layer bitmap wrong");
    require(u64(bytes.data() + 72) == 10 && u64(bytes.data() + 72 + 5 * 8) == 22,
            "Per-layer role byte metadata was normalized or dropped");
}

int main(int argc, char ** argv) {
    try {
        no_hot_allocations_and_position_rewinds();
        markers_and_full_layer_major_batches();
        overflow_and_malformed_calls_are_explicit();
        export_header_and_variable_weight_sizes(argc > 1 ? argv[1] : "expert-demand-trace-fixture.bin");
        std::cout << "logical expert demand trace framing checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        deny_allocations.store(false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
