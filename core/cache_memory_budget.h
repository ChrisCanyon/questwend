#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace questwend {

inline uint64_t parse_cache_min_free_mb(const char * value) {
    constexpr uint64_t MIB = 1024ull * 1024ull;
    if (!value) return 2560ull * MIB;
    if (!*value) throw std::invalid_argument("QWEN_CACHE_MIN_FREE_MB must be unsigned decimal MiB");
    uint64_t mib = 0;
    for (const char * cursor = value; *cursor; ++cursor) {
        if (*cursor < '0' || *cursor > '9')
            throw std::invalid_argument("QWEN_CACHE_MIN_FREE_MB must be unsigned decimal MiB");
        const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
        if (mib > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            throw std::overflow_error("QWEN_CACHE_MIN_FREE_MB exceeds unsigned integer range");
        mib = mib * 10 + digit;
    }
    if (mib > std::numeric_limits<uint64_t>::max() / MIB)
        throw std::overflow_error("QWEN_CACHE_MIN_FREE_MB exceeds byte range");
    return mib * MIB;
}

inline bool cache_headroom_exhausted(uint64_t available_bytes, uint64_t slab_bytes,
                                     uint64_t minimum_free_bytes) {
    // Explicit zero disables only this system-headroom reserve. The separate
    // resident + leased + pooled cache-capacity bounds remain authoritative.
    if (minimum_free_bytes == 0) return false;
    // Equivalent to available < minimum + slab, without overflowing addition.
    return available_bytes < minimum_free_bytes ||
           slab_bytes > available_bytes - minimum_free_bytes;
}

} // namespace questwend
