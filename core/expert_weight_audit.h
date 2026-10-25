#pragma once

// Bounded, default-off correctness instrumentation. This deliberately reads
// reference GGUF spans and can synchronize GPU copies; never benchmark it.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace questwend {

class ExpertWeightAudit {
public:
    ExpertWeightAudit(unsigned layer, unsigned skip, unsigned groups, bool gpu)
        : layer_(layer), skip_(skip), groups_(groups), gpu_(gpu) {
        if (!groups || groups > 8)
            throw std::invalid_argument("Expert byte audit groups must be within1..8");
    }
    ~ExpertWeightAudit() { for (auto & file : files_) std::fclose(file.second); }

    static std::unique_ptr<ExpertWeightAudit> from_environment(unsigned layers) {
        const char * value = std::getenv("QWEN_EXPERT_BYTE_AUDIT_LAYER");
        if (!value) return {};
        const unsigned layer = parse(value, 0);
        if (layer >= layers) throw std::invalid_argument("Expert byte audit layer is outside this model");
        const char * gpu = std::getenv("QWEN_EXPERT_BYTE_AUDIT_GPU");
        return std::make_unique<ExpertWeightAudit>(layer,
            parse(std::getenv("QWEN_EXPERT_BYTE_AUDIT_SKIP"), 0),
            parse(std::getenv("QWEN_EXPERT_BYTE_AUDIT_GROUPS"), 1),
            gpu && std::strcmp(gpu, "1") == 0);
    }

    bool begin(unsigned layer, int count, int full_count) {
        active_ = false;
        if (layer != layer_ || count != full_count || count <= 0) return false;
        const unsigned group = seen_++;
        if (group < skip_ || group - skip_ >= groups_) return false;
        active_ = true;
        current_group_ = group;
        std::fprintf(stderr, "expert-byte-audit begin layer=%u full_group=%u count=%d gpu=%d\n",
                     layer_, current_group_, count, gpu_ ? 1 : 0);
        return true;
    }
    bool active() const noexcept { return active_; }
    bool wants_gpu() const noexcept { return active_ && gpu_; }
    void end() noexcept { active_ = false; }

    void source(const std::string & path, uint64_t offset, const void * bytes, size_t size,
                int role, int expert, int slot) {
        if (active_) compare("pre-upload", path, offset, bytes, size, role, expert, slot);
    }

    template <class Readback>
    void destination(const std::string & path, uint64_t offset, size_t size,
                     int role, int expert, int slot, const Readback & readback) {
        if (!wants_gpu()) return;
        actual_.resize(size);
        readback(actual_.data(), size);
        compare("selected-GPU", path, offset, actual_.data(), size, role, expert, slot);
    }

private:
    static unsigned parse(const char * value, unsigned fallback) {
        if (!value) return fallback;
        if (!*value) throw std::invalid_argument("Expert byte audit control is empty");
        unsigned result = 0;
        for (const char * character = value; *character; ++character) {
            if (*character < '0' || *character > '9' ||
                result > (std::numeric_limits<unsigned>::max() - unsigned(*character - '0')) / 10)
                throw std::invalid_argument("Expert byte audit control must be a bounded unsigned integer");
            result = result * 10 + unsigned(*character - '0');
        }
        return result;
    }
    static uint64_t checksum(const uint8_t * data, size_t size) {
        uint64_t value = 14695981039346656037ull;
        for (size_t index = 0; index < size; ++index) { value ^= data[index]; value *= 1099511628211ull; }
        return value;
    }
    void reference(const std::string & path, uint64_t offset, size_t size) {
        FILE * file;
        auto found = files_.find(path);
        if (found == files_.end()) {
            file = std::fopen(path.c_str(), "rb");
            if (!file) throw std::runtime_error("Expert byte audit cannot open authoritative GGUF");
            try { files_.emplace(path, file); }
            catch (...) { std::fclose(file); throw; }
        } else file = found->second;
        expected_.resize(size);
#ifdef _WIN32
        const bool positioned = _fseeki64(file, static_cast<long long>(offset), SEEK_SET) == 0;
#else
        const bool positioned = fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
        if (!positioned || std::fread(expected_.data(), 1, size, file) != size)
            throw std::runtime_error("Expert byte audit reference read failed");
    }
    void compare(const char * kind, const std::string & path, uint64_t offset,
                 const void * actual, size_t size, int role, int expert, int slot) {
        if (!actual || !size) throw std::runtime_error("Expert byte audit received an empty slab");
        reference(path, offset, size);
        const auto * data = static_cast<const uint8_t *>(actual);
        const bool equal = std::memcmp(data, expected_.data(), size) == 0;
        std::fprintf(stderr,
            "expert-byte-audit %s layer=%u group=%u role=%d expert=%d slot=%d offset=%llu bytes=%zu equal=%d expected_fnv1a64=%016llx actual_fnv1a64=%016llx\n",
            kind, layer_, current_group_, role, expert, slot,
            static_cast<unsigned long long>(offset), size, equal ? 1 : 0,
            static_cast<unsigned long long>(checksum(expected_.data(), size)),
            static_cast<unsigned long long>(checksum(data, size)));
        if (!equal) {
            size_t first = 0;
            while (first < size && data[first] == expected_[first]) ++first;
            std::fprintf(stderr, "expert-byte-audit first_difference=%zu expected_byte=%u actual_byte=%u\n",
                         first, unsigned(expected_[first]), unsigned(data[first]));
            throw std::runtime_error(std::string("Authoritative GGUF byte mismatch: ") + kind);
        }
    }

    unsigned layer_, skip_, groups_, seen_ = 0, current_group_ = 0;
    bool gpu_ = false, active_ = false;
    std::map<std::string, FILE *> files_;
    std::vector<uint8_t> expected_, actual_;
};

} // namespace questwend
