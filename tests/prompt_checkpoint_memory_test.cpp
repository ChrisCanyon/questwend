// CPU-only parser and payload accounting fixture; no model/backend dependency.
#include "prompt_checkpoint_memory.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        using namespace questwend;
        require(parse_prompt_checkpoint_limit(nullptr) == 8, "Default changed");
        for (size_t expected = 0; expected <= 8; ++expected) {
            const auto text = std::to_string(expected);
            require(parse_prompt_checkpoint_limit(text.c_str()) == expected, "Valid integer changed");
        }
        require(parse_prompt_checkpoint_limit("0008") == 8, "Leading zero integer rejected");
        for (const char * text : {"", "-1", "+1", "9", "10", " 1", "1 ", "1x", "999999999999999999999"}) {
            bool rejected = false;
            try { parse_prompt_checkpoint_limit(text); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "Invalid limit accepted");
        }
        require(!should_save_prompt_checkpoint(true, 0, 100, -1), "Zero limit would copy a checkpoint");
        require(!should_save_prompt_checkpoint(false, 8, 100, -1), "Attention-only snapshot changed");
        require(!should_save_prompt_checkpoint(true, 8, 0, -1), "Empty state snapshot changed");
        require(!should_save_prompt_checkpoint(true, 8, 100, 100), "Duplicate position snapshot changed");
        require(should_save_prompt_checkpoint(true, 8, 100, 50), "Valid snapshot skipped");
        struct Snapshot { std::vector<uint8_t> blob; std::vector<float> hidden; };
        std::vector<Snapshot> checkpoints(2);
        checkpoints[0].blob.resize(123);
        checkpoints[0].blob.reserve(1000);
        checkpoints[1].blob.resize(456);
        checkpoints[1].hidden.resize(17);
        const auto memory = prompt_checkpoint_memory(checkpoints);
        require(memory.count == 2 && memory.payload_bytes == 123 + 456 + 17 * 4,
                "Payload accounting changed");
        require(memory.vector_capacity_bytes >= 1000 + 456 + 17 * 4,
                "Capacity accounting lost spare allocation");
        checkpoints.clear();
        require(prompt_checkpoint_memory(checkpoints).payload_bytes == 0,
                "Cleared checkpoints retain reported payload");
        std::cout << "Prompt checkpoint limit/skip/payload/capacity tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Prompt checkpoint unit failed: " << error.what() << '\n';
        return 1;
    }
}
