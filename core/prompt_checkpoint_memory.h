#pragma once

#include <cstddef>
#include <stdexcept>

namespace questwend {

// New explicit control only. The legacy setting is handled by the caller.
inline size_t parse_prompt_checkpoint_limit(const char * text) {
    if (!text) return 8;
    if (!*text) throw std::invalid_argument("QWEN_PROMPT_CHECKPOINT_LIMIT must be an integer within0..8");
    size_t value = 0;
    for (const char * character = text; *character; ++character) {
        if (*character < '0' || *character > '9')
            throw std::invalid_argument("QWEN_PROMPT_CHECKPOINT_LIMIT must contain decimal digits only");
        value = value * 10 + static_cast<size_t>(*character - '0');
        if (value > 8)
            throw std::invalid_argument("QWEN_PROMPT_CHECKPOINT_LIMIT must be within0..8");
    }
    return value;
}

inline bool should_save_prompt_checkpoint(bool recurrent, size_t limit,
                                         int position, int newest_position) {
    return recurrent && limit != 0 && position > 0 && position != newest_position;
}

struct PromptCheckpointMemory {
    size_t count = 0;
    size_t payload_bytes = 0;
    size_t vector_capacity_bytes = 0;
};

template<class Checkpoints>
PromptCheckpointMemory prompt_checkpoint_memory(const Checkpoints & checkpoints) {
    PromptCheckpointMemory result;
    result.count = checkpoints.size();
    for (const auto & checkpoint : checkpoints) {
        result.payload_bytes += checkpoint.blob.size() + checkpoint.hidden.size() * sizeof(float);
        result.vector_capacity_bytes += checkpoint.blob.capacity() + checkpoint.hidden.capacity() * sizeof(float);
    }
    return result;
}

} // namespace questwend
