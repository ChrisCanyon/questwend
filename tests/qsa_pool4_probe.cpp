// Model-free CUDA byte gate: old four CONT/ADD/SCALE nodes versus exact POOL2D.
// Optional raw-key file contains captured F32 rows of width128, no model loader.
#include "backend_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<uint8_t> tensor_bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static std::vector<float> corner_values(int rows) {
    const uint32_t groups[][4] = {
        {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u},
        {0u, 0x80000000u, 0u, 0x80000000u},
        {1u, 1u, 1u, 1u},
        {0x007fffffu, 0x007fffffu, 0x807fffffu, 1u},
        {0x00800000u, 0x80800000u, 1u, 0x80000001u},
        {0x7f7fffffu, 0x7f7fffffu, 0xff7fffffu, 0xff7fffffu},
        {0x7149f2cau, 0xf149f2cau, 0x40490fdbu, 0x3e000001u},
        {0x3f800001u, 0x33000001u, 0xbf800000u, 0x80000000u},
    };
    std::vector<float> values(static_cast<size_t>(rows) * 128);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < 128; ++column) {
            const uint32_t word = groups[(row / 4 + column) % 8][row % 4];
            std::memcpy(&values[static_cast<size_t>(row) * 128 + column], &word, sizeof(word));
        }
    return values;
}

static std::vector<float> finite_values(int rows) {
    std::vector<float> values(static_cast<size_t>(rows) * 128);
    uint32_t state = 0x260124u;
    for (float & value : values) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<float>(static_cast<int>((state >> 8) % 20001u) - 10000) / 4096.0f;
    }
    return values;
}

static void run_case(ggml_backend_t backend, const std::vector<float> & input,
                     const char * label, bool generic_only = false) {
    check(!input.empty() && input.size() % (128 * 4) == 0, "input requires full four-row blocks");
    const int rows = static_cast<int>(input.size() / 128);
    const int blocks = rows / 4;
    ggml_init_params params{};
    params.no_alloc = true;
    params.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false);
    ggml_context * state_context = ggml_init(params);
    ggml_context * context = ggml_init(params);
    check(state_context && context, "context allocation failed");
    ggml_tensor * source = ggml_new_tensor_2d(state_context, GGML_TYPE_F32, 128, rows);
    ggml_backend_buffer_t source_buffer = ggml_backend_alloc_ctx_tensors(state_context, backend);
    check(source_buffer != nullptr, "source allocation failed");
    ggml_tensor * members = ggml_reshape_3d(context, source, 128, 4, blocks);
    ggml_tensor * old = nullptr;
    for (int row = 0; row < 4; ++row) {
        auto * slice = ggml_cont(context, ggml_view_2d(context, members, 128, blocks,
            members->nb[2], static_cast<size_t>(row) * members->nb[1]));
        old = old ? ggml_add(context, old, slice) : slice;
    }
    old = ggml_scale(context, old, 0.25f);
    ggml_set_output(old);
    ggml_tensor * candidate = ggml_pool_2d(context, source, GGML_OP_POOL_AVG,
        1, 4, 1, 4, 0.0f, 0.0f);
    // Untagged control must keep the generic AVG semantics even with flag1.
    ggml_set_name(candidate, generic_only ? "ordinary_pool_avg_control" : "qsa_pool4_exact_fixture");
    ggml_set_output(candidate);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 128, false);
    ggml_build_forward_expand(graph, old);
    ggml_build_forward_expand(graph, candidate);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    check(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    bool observed_generic_difference = false;
    for (int replay = 0; replay < 5; ++replay) {
        // Same bytes, changing source address contents between captured replays.
        std::vector<float> values = input;
        if (replay % 2 && !generic_only) {
            for (size_t index = 0; index < values.size(); index += 19) values[index] = -values[index];
        }
        ggml_backend_tensor_set(source, values.data(), 0, values.size() * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        const bool exact = tensor_bytes(old) == tensor_bytes(candidate);
        if (generic_only) observed_generic_difference = observed_generic_difference || !exact;
        else check(exact, "exact POOL4 differs from original GPU add/scale bytes");
    }
    if (generic_only)
        check(observed_generic_difference, "corner control did not distinguish generic AVG semantics");
    std::cout << "pool4 case=" << label << " rows=" << rows
              << " exact=" << !generic_only << " generic_control=" << generic_only << '\n';
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(source_buffer);
    ggml_free(context);
    ggml_free(state_context);
}

static std::vector<float> read_raw(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    check(static_cast<bool>(input), "cannot open copied raw-key fixture");
    const std::streamoff count = input.tellg();
    check(count > 0 && count <= static_cast<std::streamoff>(128ull * 262144 * sizeof(float)) &&
        count % static_cast<std::streamoff>(128 * 4 * sizeof(float)) == 0,
        "raw-key fixture must be bounded F32 width128 with full four-row blocks");
    std::vector<float> values(static_cast<size_t>(count) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count));
    check(static_cast<bool>(input), "truncated copied raw-key fixture");
    return values;
}

int main(int argc, char ** argv) {
    ggml_backend_t backend = nullptr;
    try {
        check(argc == 2 || argc == 3,
            "Usage: qw-qsa-pool4-probe <matched ggml-cuda.dll> [copied-real-qsa-raw.f32]");
#ifdef _WIN32
        _putenv_s("QW_CUDA_BACKEND", argv[1]);
        _putenv_s("GGML_CUDA_QSA_POOL4_EXACT", "1");
#else
        setenv("QW_CUDA_BACKEND", argv[1], 1);
        setenv("GGML_CUDA_QSA_POOL4_EXACT", "1", 1);
#endif
        questwend::initialize_backends();
        auto * registry = ggml_backend_reg_by_name("CUDA");
        check(registry != nullptr, "matched CUDA backend unavailable");
        backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
        check(backend != nullptr, "CUDA initialization failed");
        for (int rows : {4, 8, 28, 8192}) {
            run_case(backend, finite_values(rows), "finite");
            run_case(backend, corner_values(rows), "edge-bits");
        }
        run_case(backend, corner_values(28), "untagged-AVG-must-stay-generic", true);
        if (argc == 3) run_case(backend, read_raw(argv[2]), "copied-real-QSA");
        ggml_backend_free(backend);
        std::cout << "exact QSA pool4 GPU byte gates passed; no throughput claim\n";
        return 0;
    } catch (const std::exception & error) {
        if (backend) ggml_backend_free(backend);
        std::cerr << "exact QSA pool4 gate failed: " << error.what() << '\n';
        return 1;
    }
}
