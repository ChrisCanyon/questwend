#pragma once

// Optional logical routing observation, never a routing/cache decision.
// Allocate once at load. Record calls only write within the existing buffer;
// file I/O is confined to an explicit save/profile boundary.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace questwend {

class ExpertDemandTrace {
public:
    enum class Phase : uint16_t { single_forward = 1, prefill_batch = 2, verify_batch = 3 };
    static constexpr size_t file_header_bytes = 64, record_header_bytes = 32;
    static constexpr uint16_t complete = 1, marker_only = 2, aborted = 4, invalid = 8;

    class Call {
    public:
        Call() noexcept = default;
        Call(const Call &) = delete;
        Call & operator=(const Call &) = delete;
        Call(Call && other) noexcept
            : owner_(other.owner_), offset_(other.offset_), seen_(other.seen_),
              rows_(other.rows_), flags_(other.flags_), exceptions_(other.exceptions_) {
            other.owner_ = nullptr;
        }
        ~Call() noexcept {
            if (owner_) owner_->finish(*this, std::uncaught_exceptions() > exceptions_);
        }
        void layer(int layer, const int32_t * ids, int rows, int n_used) noexcept {
            if (owner_) owner_->record_layer(*this, layer, ids, rows, n_used);
        }

    private:
        friend class ExpertDemandTrace;
        Call(ExpertDemandTrace * owner, size_t offset, uint32_t rows, uint16_t flags) noexcept
            : owner_(owner), offset_(offset), rows_(rows), flags_(flags),
              exceptions_(std::uncaught_exceptions()) {}
        ExpertDemandTrace * owner_ = nullptr;
        size_t offset_ = 0;
        uint64_t seen_ = 0;
        uint32_t rows_ = 0;
        uint16_t flags_ = 0;
        int exceptions_ = 0;
    };

    ExpertDemandTrace(uint16_t layers, uint16_t experts, uint16_t n_used,
                      std::vector<uint64_t> role_bytes, size_t capacity = 16u << 20,
                      bool include_prefill = false)
        : layers_(layers), experts_(experts), n_used_(n_used),
          role_bytes_(std::move(role_bytes)), storage_(capacity, 0),
          include_prefill_(include_prefill) {
        if (!layers || layers > 64 || !experts || !n_used || n_used > experts ||
            role_bytes_.size() != static_cast<size_t>(layers) * 3 || capacity < record_header_bytes)
            throw std::invalid_argument("Unsupported logical demand trace dimensions/capacity");
        for (uint16_t layer = 0; layer < layers_; ++layer) {
            const auto offset = static_cast<size_t>(layer) * 3;
            const bool gate = role_bytes_[offset] != 0, up = role_bytes_[offset + 1] != 0,
                       down = role_bytes_[offset + 2] != 0;
            if (gate != up || gate != down)
                throw std::invalid_argument("Logical trace requires whole-expert byte metadata");
            if (gate) expected_layers_ |= uint64_t{1} << layer;
        }
    }

    Call begin(int position, int rows, Phase phase) noexcept {
        const uint64_t call_id = next_call_++;
        // Preserve one contiguous chronological prefix after capacity exhaustion.
        // Later smaller calls must not silently fill holes after a dropped call.
        if (truncated_) { ++dropped_; return {}; }
        const bool is_marker = phase == Phase::prefill_batch && !include_prefill_;
        if (active_ || position < 0 || rows < 1 ||
            (phase != Phase::single_forward && phase != Phase::prefill_batch && phase != Phase::verify_batch)) {
            invalid_ = true;
            ++dropped_;
            return {};
        }
        const uint64_t payload = is_marker ? 0 :
            static_cast<uint64_t>(layers_) * static_cast<uint32_t>(rows) * n_used_ * 2;
        if (payload > std::numeric_limits<uint32_t>::max() ||
            payload > storage_.size() - used_ || record_header_bytes > storage_.size() - used_ - payload) {
            truncated_ = true;
            ++dropped_;
            return {};
        }
        const size_t offset = used_;
        used_ += record_header_bytes + static_cast<size_t>(payload);
        auto * record = storage_.data() + offset;
        write64(record, call_id);
        write32(record + 8, static_cast<uint32_t>(position));
        write32(record + 12, static_cast<uint32_t>(rows));
        write16(record + 16, static_cast<uint16_t>(phase));
        write16(record + 18, layers_);
        write16(record + 20, n_used_);
        const uint16_t flags = is_marker ? marker_only : 0;
        write16(record + 22, flags);
        write32(record + 24, static_cast<uint32_t>(payload));
        write32(record + 28, 0);
        active_ = true;
        return Call(this, offset, static_cast<uint32_t>(rows), flags);
    }

    bool save(const std::string & path) const {
        if (active_) return false; // An export must be outside active inference.
        FILE * file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        uint8_t header[file_header_bytes]{};
        const char magic[8] = {'Q', 'W', 'D', 'E', 'M', 'T', 'R', '1'};
        for (size_t index = 0; index < 8; ++index) header[index] = static_cast<uint8_t>(magic[index]);
        write32(header + 8, 1);
        write32(header + 12, 0x01020304);
        write64(header + 16, storage_.size());
        write64(header + 24, used_);
        write64(header + 32, next_call_);
        write64(header + 40, dropped_);
        write32(header + 48, (truncated_ ? 1u : 0u) | (invalid_ ? 2u : 0u));
        write16(header + 52, layers_);
        write16(header + 54, experts_);
        write16(header + 56, n_used_);
        write16(header + 58, 0);
        write32(header + 60, 8u + static_cast<uint32_t>(role_bytes_.size() * 8));
        bool ok = std::fwrite(header, 1, sizeof(header), file) == sizeof(header);
        uint8_t value[8];
        write64(value, expected_layers_);
        ok = std::fwrite(value, 1, sizeof(value), file) == sizeof(value) && ok;
        for (uint64_t bytes : role_bytes_) {
            write64(value, bytes);
            ok = std::fwrite(value, 1, sizeof(value), file) == sizeof(value) && ok;
        }
        ok = std::fwrite(storage_.data(), 1, used_, file) == used_ && ok;
        return std::fclose(file) == 0 && ok;
    }

    size_t used_bytes() const noexcept { return used_; }
    bool truncated() const noexcept { return truncated_; }
    bool invalid_trace() const noexcept { return invalid_; }
    uint64_t dropped_calls() const noexcept { return dropped_; }
    const uint8_t * records() const noexcept { return storage_.data(); }

private:
    static void write16(uint8_t * destination, uint16_t value) noexcept {
        destination[0] = static_cast<uint8_t>(value);
        destination[1] = static_cast<uint8_t>(value >> 8);
    }
    static void write32(uint8_t * destination, uint32_t value) noexcept {
        for (size_t byte = 0; byte < 4; ++byte) destination[byte] = static_cast<uint8_t>(value >> (byte * 8));
    }
    static void write64(uint8_t * destination, uint64_t value) noexcept {
        for (size_t byte = 0; byte < 8; ++byte) destination[byte] = static_cast<uint8_t>(value >> (byte * 8));
    }
    void record_layer(Call & call, int layer, const int32_t * ids, int rows, int n_used) noexcept {
        if (call.flags_ & marker_only) return;
        if (layer < 0 || layer >= layers_ || !ids || rows != static_cast<int>(call.rows_) || n_used != n_used_ ||
            !(expected_layers_ & (uint64_t{1} << layer)) || (call.seen_ & (uint64_t{1} << layer))) {
            call.flags_ |= invalid;
            return;
        }
        const size_t count = static_cast<size_t>(rows) * n_used_;
        uint8_t * destination = storage_.data() + call.offset_ + record_header_bytes +
            static_cast<size_t>(layer) * count * 2;
        for (size_t index = 0; index < count; ++index) {
            if (ids[index] < 0 || ids[index] >= experts_) { call.flags_ |= invalid; return; }
            write16(destination + index * 2, static_cast<uint16_t>(ids[index]));
        }
        call.seen_ |= uint64_t{1} << layer;
    }
    void finish(Call & call, bool unwinding) noexcept {
        if (unwinding) call.flags_ |= aborted;
        if (!(call.flags_ & marker_only) && call.seen_ != expected_layers_) call.flags_ |= invalid;
        if (call.flags_ & invalid) invalid_ = true;
        if (!(call.flags_ & (invalid | aborted))) call.flags_ |= complete;
        write16(storage_.data() + call.offset_ + 22, call.flags_);
        active_ = false;
    }

    uint16_t layers_, experts_, n_used_;
    std::vector<uint64_t> role_bytes_;
    std::vector<uint8_t> storage_;
    uint64_t expected_layers_ = 0, next_call_ = 0, dropped_ = 0;
    size_t used_ = 0;
    bool include_prefill_ = false, active_ = false, truncated_ = false, invalid_ = false;
};

} // namespace questwend
