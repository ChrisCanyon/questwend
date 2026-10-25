#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <exception>
#include <random>
#include "ggml-backend.h"
#include "expert_cost_stats.h"

namespace expert_cost {
struct Slab {
    std::string path;
    uint64_t offset = 0;
    size_t bytes = 0;
    int type = 0;
    int64_t columns = 0, rows = 0;
};
using Expert = std::vector<Slab>;

// One reusable staging allocation. Pinned allocation failure is a failed
// pinned case, rather than silently changing the tested memory type.
class HostBuffer {
    ggml_backend_buffer_t pinned_ = nullptr;
    std::vector<uint8_t> ordinary_;
public:
    uint8_t * data = nullptr;
    size_t bytes = 0;
    HostBuffer(size_t size, ggml_backend_buffer_type_t pinned_type = nullptr): bytes(size) {
        if (pinned_type) {
            pinned_ = ggml_backend_buft_alloc_buffer(pinned_type, size);
            if (!pinned_) throw std::runtime_error("Pinned staging allocation failed");
            data = (uint8_t *) ggml_backend_buffer_get_base(pinned_);
        } else { ordinary_.resize(size); data = ordinary_.data(); }
    }
    ~HostBuffer() { if (pinned_) ggml_backend_buffer_free(pinned_); }
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer & operator=(const HostBuffer &) = delete;
};

// Mirrors expert_cache.cpp's aligned Windows direct-read path. Bypasses the
// Windows file cache, not the SSD controller's cache. The aligned bounce copy
// is inside the measured read path, and actual aligned bytes are reported.
class FileReader {
    bool direct_;
    std::map<std::string, HANDLE> handles_;
    uint8_t * aligned_ = nullptr;
    size_t capacity_ = 0;
public:
    uint64_t actual_bytes = 0;
    explicit FileReader(bool direct): direct_(direct) {}
    ~FileReader() {
        for (auto & item : handles_) CloseHandle(item.second);
        if (aligned_) _aligned_free(aligned_);
    }
    void read(const Slab & slab, void * destination) {
        auto found = handles_.find(slab.path);
        if (found == handles_.end()) {
            HANDLE handle = CreateFileA(slab.path.c_str(), GENERIC_READ,
                FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                direct_ ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_RANDOM_ACCESS, nullptr);
            if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open GGUF read handle");
            found = handles_.emplace(slab.path, handle).first;
        }
        const uint64_t offset = direct_ ? slab.offset & ~uint64_t(4095) : slab.offset;
        const size_t size = direct_ ? (size_t) (((slab.offset + slab.bytes + 4095) & ~uint64_t(4095)) - offset) : slab.bytes;
        void * target = destination;
        if (direct_) {
            if (size > capacity_) {
                if (aligned_) _aligned_free(aligned_);
                aligned_ = (uint8_t *) _aligned_malloc(size, 4096);
                if (!aligned_) throw std::bad_alloc();
                capacity_ = size;
            }
            target = aligned_;
        }
        OVERLAPPED positioned{};
        positioned.Offset = (DWORD) offset;
        positioned.OffsetHigh = (DWORD) (offset >> 32);
        DWORD got = 0;
        if (!ReadFile(found->second, target, (DWORD) size, &got, &positioned) ||
            got + offset < slab.offset + slab.bytes)
            throw std::runtime_error("Expert read failed or returned incomplete bytes");
        actual_bytes += got;
        if (direct_) std::memcpy(destination, aligned_ + slab.offset - offset, slab.bytes);
    }
};

class ExpertReader {
    std::vector<std::unique_ptr<FileReader>> workers_;
public:
    explicit ExpertReader(bool direct, int workers) {
        for (int i = 0; i < workers; ++i) workers_.emplace_back(new FileReader(direct));
    }
    uint64_t actual_bytes() const {
        uint64_t result = 0;
        for (auto & worker : workers_) result += worker->actual_bytes;
        return result;
    }
    void read(const std::vector<Expert> & corpus, const std::vector<int> & selected,
              const std::vector<size_t> & role_offsets, uint8_t * destination) {
        const int count = std::min((int) workers_.size(), (int) selected.size());
        std::vector<std::exception_ptr> errors(count);
        auto work = [&](int thread) {
            try {
                for (int index = thread; index < (int) selected.size(); index += count) {
                    const auto & expert = corpus[selected[index]];
                    for (int role = 0; role < 3; ++role)
                        workers_[thread]->read(expert[role], destination + role_offsets[role] + index * expert[role].bytes);
                }
            } catch (...) { errors[thread] = std::current_exception(); }
        };
        std::vector<std::thread> threads;
        for (int i = 1; i < count; ++i) threads.emplace_back(work, i);
        work(0);
        for (auto & thread : threads) thread.join();
        for (auto & error : errors) if (error) std::rethrow_exception(error);
    }
};
}
