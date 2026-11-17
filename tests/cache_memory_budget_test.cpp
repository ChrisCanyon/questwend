#include "../core/cache_memory_budget.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace questwend;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
static void rejects(Function function, const char * message) {
    bool failed = false;
    try { function(); } catch (const std::exception &) { failed = true; }
    check(failed, message);
}

int main() {
    constexpr uint64_t MIB = 1024ull * 1024ull;
    constexpr uint64_t MAX = std::numeric_limits<uint64_t>::max();
    const uint64_t baseline = 2560ull * MIB;
    check(parse_cache_min_free_mb(nullptr) == baseline, "Unset reserve changed");
    check(parse_cache_min_free_mb("2560") == baseline, "Explicit default differs");
    check(parse_cache_min_free_mb("0") == 0, "Zero reserve not accepted");
    check(parse_cache_min_free_mb("000") == 0, "Unsigned zero spelling rejected");
    check(parse_cache_min_free_mb("17") == 17 * MIB, "MiB conversion failed");
    check(parse_cache_min_free_mb("00017") == 17 * MIB, "Leading zero decimal failed");
    for (const char * invalid : {"", "-1", "+1", " 1", "1 ", "1\n", "1.0", "1G", "0x10", "12x"})
        rejects([&] { parse_cache_min_free_mb(invalid); }, "Malformed reserve accepted");
    const auto maximum_mib = std::to_string(MAX / MIB);
    const auto excessive_mib = std::to_string(MAX / MIB + 1);
    check(parse_cache_min_free_mb(maximum_mib.c_str()) == (MAX / MIB) * MIB,
          "Largest byte-representable MiB reserve rejected");
    rejects([&] { parse_cache_min_free_mb(excessive_mib.c_str()); }, "MiB overflow accepted");
    rejects([&] { parse_cache_min_free_mb("18446744073709551615"); }, "Maximum uint64 MiB overflow accepted");
    rejects([&] { parse_cache_min_free_mb("18446744073709551616"); }, "Integer overflow accepted");
    rejects([&] { parse_cache_min_free_mb("99999999999999999999999999999999999"); }, "Long overflow accepted");

    check(!cache_headroom_exhausted(0, MAX, 0), "Zero reserve did not disable guard");
    check(!cache_headroom_exhausted(MAX, MAX, 0), "Zero reserve inspected slab size");
    check(cache_headroom_exhausted(baseline - 1, 0, baseline), "Below reserve accepted");
    check(!cache_headroom_exhausted(baseline, 0, baseline), "Exact reserve rejected");
    check(cache_headroom_exhausted(baseline, 1, baseline), "Slab reserve omitted");
    check(!cache_headroom_exhausted(baseline + MIB, MIB, baseline), "Exact old boundary rejected");
    check(cache_headroom_exhausted(baseline + MIB - 1, MIB, baseline), "Below old boundary accepted");
    check(!cache_headroom_exhausted(MAX, 0, MAX), "Maximum exact reserve failed");
    check(cache_headroom_exhausted(MAX, 1, MAX), "Reserve addition overflowed");
    check(cache_headroom_exhausted(MAX, MAX, baseline), "Large slab addition overflowed");
    for (uint64_t available : {0ull, baseline - 1, baseline, baseline + MIB, 32768ull * MIB})
        for (uint64_t bytes : {0ull, 1ull, MIB, 64ull * MIB})
            check(cache_headroom_exhausted(available, bytes, baseline) ==
                  (available < baseline + bytes), "Default guard differs from old expression");
    std::cout << "cache memory budget parser and decision gates passed\n";
}
