#pragma once

// Unchanged production covering-range read, shared with CPU-only tests.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace questwend {

// A separate guarded prototype. The original bounce reader below is unchanged.
struct DirectCoveringRead {
    uint64_t file_offset = 0;
    size_t read_bytes = 0, payload_delta = 0, payload_bytes = 0;
};

inline bool direct_covering_range(uint64_t offset, size_t bytes, DirectCoveringRead & plan) {
    constexpr uint64_t sector = 4096;
    constexpr uint64_t maximum = UINT64_MAX;
    if (!bytes || bytes > UINT32_MAX - sector || offset > maximum - bytes ||
        offset + bytes > maximum - (sector - 1)) return false;
    const uint64_t start = offset & ~(sector - 1);
    const uint64_t end = (offset + bytes + sector - 1) & ~(sector - 1);
    const uint64_t count = end - start;
    if (!count || count > UINT32_MAX) return false;
    plan = {start, static_cast<size_t>(count), static_cast<size_t>(offset - start), bytes};
    return true;
}

inline bool direct_covering_stride(size_t maximum_payload, size_t & stride) {
    DirectCoveringRead worst;
    if (!direct_covering_range(4095, maximum_payload, worst)) return false;
    stride = worst.read_bytes;
    return true;
}

// Read the aligned covering range directly into its own pinned staging slice.
// Payload begins at destination+payload_delta; no CPU bounce memcpy is needed.
inline bool win_direct_read_covering(void * handle, void * destination, size_t capacity,
                                    const DirectCoveringRead & plan) {
    constexpr size_t sector = 4096;
    if (!handle || !destination || (reinterpret_cast<uintptr_t>(destination) & (sector - 1)) ||
        (plan.file_offset & (sector - 1)) || !plan.read_bytes || (plan.read_bytes & (sector - 1)) ||
        plan.file_offset > UINT64_MAX - plan.read_bytes ||
        plan.read_bytes > capacity || plan.read_bytes > UINT32_MAX || plan.payload_delta >= sector ||
        plan.payload_delta > plan.read_bytes || !plan.payload_bytes ||
        plan.payload_bytes > plan.read_bytes - plan.payload_delta) return false;
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(plan.file_offset & 0xffffffffull);
    operation.OffsetHigh = static_cast<DWORD>(plan.file_offset >> 32);
    DWORD received = 0;
    if (!ReadFile(static_cast<HANDLE>(handle), destination, static_cast<DWORD>(plan.read_bytes),
                  &received, &operation)) return false;
    return received >= plan.payload_delta + plan.payload_bytes;
}

// Positioned read with FILE_FLAG_NO_BUFFERING semantics: offset, length and
// buffer must be sector-aligned, so read the covering aligned range into an
// (over-allocated) aligned staging vector and copy the slab out. Bypasses the
// OS page cache — true SSD reads even when the GGUF fits in RAM.
inline bool win_direct_read(void * h, void * dst, size_t nbytes, uint64_t off,
                            std::vector<uint8_t> & abuf) {
    const uint64_t SEC = 4096;
    const uint64_t a0   = off & ~(SEC - 1);
    const uint64_t a1   = (off + nbytes + SEC - 1) & ~(SEC - 1);
    const size_t   alen = (size_t) (a1 - a0);
    if (abuf.size() < alen + SEC) abuf.resize(alen + SEC);
    uint8_t * ap = (uint8_t *) (((uintptr_t) abuf.data() + SEC - 1) & ~(uintptr_t) (SEC - 1));
    OVERLAPPED ov{};
    ov.Offset     = (DWORD) (a0 & 0xffffffffull);
    ov.OffsetHigh = (DWORD) (a0 >> 32);
    DWORD got = 0;
    if (!ReadFile((HANDLE) h, ap, (DWORD) alen, &got, &ov)) return false;
    if ((uint64_t) got + a0 < off + nbytes) return false;   // short read (EOF)
    memcpy(dst, ap + (off - a0), nbytes);
    return true;
}

inline void * win_direct_open(const std::string & path) {
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    return h == INVALID_HANDLE_VALUE ? nullptr : (void *) h;
}

} // namespace questwend
#endif
