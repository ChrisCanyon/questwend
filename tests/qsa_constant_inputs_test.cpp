#include "qsa_constant_inputs.h"

#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

using questwend::QsaConstantInputs;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void compare_original_values(int blocks, int cells_per_block) {
    // Original set_qsa_inputs loops, kept as the byte-level reference.
    std::vector<int32_t> positions(static_cast<size_t>(4) * blocks);
    for (int section = 0; section < 4; ++section)
        for (int block = 0; block < blocks; ++block)
            positions[static_cast<size_t>(section) * blocks + block] = block * cells_per_block;
    std::vector<int32_t> cells(static_cast<size_t>(cells_per_block) * blocks);
    for (int block = 0; block < blocks; ++block)
        for (int cell = 0; cell < cells_per_block; ++cell)
            cells[static_cast<size_t>(block) * cells_per_block + cell] = block * cells_per_block + cell;
    QsaConstantInputs cache;
    check(cache.needs_upload(blocks, cells_per_block), "fresh graph must initialize");
    cache.prepare(blocks, cells_per_block);
    check(cache.positions().size() == positions.size() && cache.cells().size() == cells.size(), "shape differs");
    check(std::memcmp(cache.positions().data(), positions.data(), positions.size() * sizeof(int32_t)) == 0,
          "position bytes differ");
    check(std::memcmp(cache.cells().data(), cells.data(), cells.size() * sizeof(int32_t)) == 0,
          "cell bytes differ");
    cache.mark_uploaded();
    check(!cache.needs_upload(blocks, cells_per_block), "same graph should reuse constants");
    check(cache.needs_upload(blocks + 1, cells_per_block), "bucket shape change needs upload");
    check(cache.needs_upload(blocks, cells_per_block + 1), "block width change needs upload");
    cache.prepare(blocks + 1, cells_per_block);
    check(cache.needs_upload(blocks + 1, cells_per_block), "prepare is not a successful upload");
    cache.mark_uploaded();
    check(!cache.needs_upload(blocks + 1, cells_per_block), "new shape upload not recorded");
}

static void recycled_owner_address_does_not_reuse_upload_state() {
    alignas(QsaConstantInputs) unsigned char storage[sizeof(QsaConstantInputs)];
    auto * first_graph = new (storage) QsaConstantInputs;
    first_graph->prepare(64, 8);
    first_graph->mark_uploaded();
    check(!first_graph->needs_upload(64, 8), "initial setup failed");
    first_graph->~QsaConstantInputs();
    // Poison the exact same address to catch dependence on prior object bytes.
    std::memset(storage, 0xa5, sizeof(storage));
    auto * replacement_graph = new (storage) QsaConstantInputs;
    check(replacement_graph->needs_upload(64, 8), "recycled graph identity retained stale upload state");
    replacement_graph->prepare(64, 8);
    replacement_graph->mark_uploaded();
    replacement_graph->~QsaConstantInputs();
}

static void opt_in_environment() {
    const char * name = "QWEN_QSA_CONSTANT_INPUT_CACHE";
    const char * previous = std::getenv(name);
    const bool existed = previous != nullptr;
    const std::string saved = previous ? previous : "";
    auto set = [&](const char * value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) setenv(name, value, 1); else unsetenv(name);
#endif
    };
    set(nullptr);
    check(!QsaConstantInputs::enabled_from_environment(), "unset must preserve old path");
    set("0");
    check(!QsaConstantInputs::enabled_from_environment(), "zero must preserve old path");
    set("1");
    check(QsaConstantInputs::enabled_from_environment(), "one should opt in");
    set(existed ? saved.c_str() : nullptr);
}

int main() {
    try {
        compare_original_values(1, 8);
        compare_original_values(64, 8);
        compare_original_values(4096, 16);
        compare_original_values(8192, 16);
        recycled_owner_address_does_not_reuse_upload_state();
        opt_in_environment();
        std::cout << "QSA constant input tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "QSA constant input test failed: " << error.what() << '\n';
        return 1;
    }
}
