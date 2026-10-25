#include "expert_ssd_windows.h"
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace questwend;

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    void * memory = nullptr;
    HANDLE reader = INVALID_HANDLE_VALUE;
    std::filesystem::path filename;
    try {
        std::array<char, MAX_PATH> temporary{};
        check(GetTempPathA(static_cast<DWORD>(temporary.size()), temporary.data()) > 0, "GetTempPath failed");
        std::array<char, MAX_PATH> name{};
        check(GetTempFileNameA(temporary.data(), "qwc", 0, name.data()) != 0, "GetTempFileName failed");
        filename = name.data();
        const size_t source_bytes = 8u << 20;
        std::vector<uint8_t> source(source_bytes);
        for (size_t index = 0; index < source.size(); ++index)
            source[index] = static_cast<uint8_t>((index * 37 + index / 4096 * 11) & 255);
        HANDLE output = CreateFileA(name.data(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(output != INVALID_HANDLE_VALUE, "Cannot create synthetic source");
        DWORD written = 0;
        const BOOL write_ok = WriteFile(output, source.data(), static_cast<DWORD>(source.size()), &written, nullptr);
        CloseHandle(output);
        check(write_ok && written == source.size(), "Synthetic source write failed");
        reader = static_cast<HANDLE>(win_direct_open(filename.string()));
        check(reader && reader != INVALID_HANDLE_VALUE, "Cannot open direct synthetic reader");
        size_t stride = 0;
        check(direct_covering_stride(1740800, stride), "Real maximum stride rejected");
        const size_t allocation = stride * 3 + 8192;
        memory = VirtualAlloc(nullptr, allocation, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        check(memory != nullptr, "Cannot allocate aligned synthetic staging");
        auto * base = static_cast<uint8_t *>(memory) + 4096;
        std::vector<uint8_t> bounce;
        int cases = 0;
        for (size_t bytes : {size_t(921600), size_t(1740800), size_t(1126400)}) {
            for (uint64_t offset : {0ull, 32ull, 672ull, 4095ull, 4096ull}) {
                DirectCoveringRead plan;
                check(direct_covering_range(offset, bytes, plan), "Real covering plan rejected");
                std::memset(memory, 0xA5, allocation);
                check(win_direct_read_covering(reader, base, stride, plan), "Covering read failed");
                std::vector<uint8_t> old(bytes);
                check(win_direct_read(reader, old.data(), bytes, offset, bounce), "Old fallback read failed");
                check(std::memcmp(base + plan.payload_delta, old.data(), bytes) == 0, "Old and new payload differ");
                check(std::memcmp(base + plan.payload_delta, source.data() + offset, bytes) == 0, "Wrong authoritative bytes");
                for (size_t index = 0; index < 4096; ++index)
                    check(static_cast<uint8_t *>(memory)[index] == 0xA5, "Pre-range sentinel changed");
                for (size_t index = plan.read_bytes; index < stride * 3 + 4096; ++index)
                    check(base[index] == 0xA5, "Read escaped its covering range");
                check(!win_direct_read_covering(reader, base + 1, stride, plan), "Misaligned destination accepted");
                check(!win_direct_read_covering(reader, base, plan.read_bytes - 1, plan), "Undersized slice accepted");
                ++cases;
            }
        }
        // Distinct role slices must survive subsequent reads without overlap.
        std::memset(memory, 0xA5, allocation);
        std::array<DirectCoveringRead, 3> plans;
        const std::array<size_t, 3> sizes{{921600, 921600, 1740800}};
        const std::array<uint64_t, 3> offsets{{672, 2u << 20, (4u << 20) + 32}};
        for (size_t index = 0; index < plans.size(); ++index) {
            check(direct_covering_range(offsets[index], sizes[index], plans[index]), "Multi-role plan failed");
            check(win_direct_read_covering(reader, base + index * stride, stride, plans[index]), "Multi-role read failed");
        }
        for (size_t index = 0; index < plans.size(); ++index)
            check(std::memcmp(base + index * stride + plans[index].payload_delta,
                              source.data() + offsets[index], sizes[index]) == 0, "A later role overwrote an earlier payload");
        DirectCoveringRead invalid;
        check(!direct_covering_range(0, 0, invalid), "Zero-size payload accepted");
        check(!direct_covering_range(UINT64_MAX - 10, 32, invalid), "Offset overflow accepted");
        check(!direct_covering_range(0, static_cast<size_t>(UINT32_MAX), invalid), "DWORD overflow accepted");
        size_t bad_stride = 0;
        check(!direct_covering_stride(0, bad_stride), "Empty stride accepted");
        VirtualFree(memory, 0, MEM_RELEASE); memory = nullptr;
        CloseHandle(reader); reader = INVALID_HANDLE_VALUE;
        std::filesystem::remove(filename);
        std::cout << "Direct pinned covering: " << cases << " payload comparisons, sentinels, slice isolation and bounds passed\n";
        return 0;
    } catch (const std::exception & error) {
        if (memory) VirtualFree(memory, 0, MEM_RELEASE);
        if (reader && reader != INVALID_HANDLE_VALUE) CloseHandle(reader);
        if (!filename.empty()) std::filesystem::remove(filename);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
