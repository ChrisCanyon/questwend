#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace questwend {

// Owned by one HyGraph, never a process-global map keyed by recycled pointers.
// Destroying/replacing that graph also destroys/replaces its upload state.
class QsaConstantInputs {
public:
    static bool enabled_from_environment() {
        const char * value = std::getenv("QWEN_QSA_CONSTANT_INPUT_CACHE");
        return value && std::strcmp(value, "1") == 0;
    }

    bool needs_upload(int blocks, int cells_per_block) const {
        validate(blocks, cells_per_block);
        return !uploaded_ || blocks != blocks_ || cells_per_block != cells_per_block_;
    }

    void prepare(int blocks, int cells_per_block) {
        validate(blocks, cells_per_block);
        blocks_ = blocks;
        cells_per_block_ = cells_per_block;
        uploaded_ = false;
        positions_.resize(static_cast<size_t>(4) * blocks);
        for (int section = 0; section < 4; ++section)
            for (int block = 0; block < blocks; ++block)
                positions_[static_cast<size_t>(section) * blocks + block] = block * cells_per_block;
        cells_.resize(static_cast<size_t>(cells_per_block) * blocks);
        for (int block = 0; block < blocks; ++block)
            for (int cell = 0; cell < cells_per_block; ++cell)
                cells_[static_cast<size_t>(block) * cells_per_block + cell] = block * cells_per_block + cell;
    }

    void mark_uploaded() {
        if (positions_.empty() || cells_.empty())
            throw std::logic_error("QSA constants were not prepared");
        uploaded_ = true;
    }

    const std::vector<int32_t> & positions() const { return positions_; }
    const std::vector<int32_t> & cells() const { return cells_; }

private:
    static void validate(int blocks, int cells_per_block) {
        if (blocks <= 0 || cells_per_block <= 0 ||
            static_cast<int64_t>(blocks) * cells_per_block > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument("Invalid QSA constant input shape");
    }

    int blocks_ = 0, cells_per_block_ = 0;
    bool uploaded_ = false;
    std::vector<int32_t> positions_, cells_;
};

} // namespace questwend
