// Model-free CUDA gather against authoritative CPU indexed-copy bytes.
// Run in separate processes with candidate flag0/1; no timing claim here.
#include "backend_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static uint32_t value_bits(int index) {
    const uint32_t special[] = {0, 0x80000000u, 0x3f800000u, 0xbf800000u,
        0x7fc01234u, 0x7f800000u, 0xff800000u, 0x00000001u};
    return index % 3 ? (0x3f000000u + static_cast<uint32_t>(index)) : special[(index / 3) % 8];
}

static void run_case(ggml_backend_t backend, int count, int planes, int batches, bool padded) {
    const int rows = 67;
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
    params.no_alloc = true;
    ggml_context * values_ctx = ggml_init(params);
    ggml_context * graph_ctx = ggml_init(params);
    check(values_ctx && graph_ctx, "context allocation failed");
    ggml_tensor * source_base = ggml_new_tensor_4d(values_ctx, GGML_TYPE_F32,
        padded ? 4 : 1, rows, planes, batches);
    ggml_tensor * source = padded ? ggml_view_4d(values_ctx, source_base, 1, rows, planes, batches,
        source_base->nb[1], source_base->nb[2], source_base->nb[3], sizeof(float)) : source_base;
    ggml_tensor * index_base = ggml_new_tensor_3d(values_ctx, GGML_TYPE_I32,
        padded ? count * 2 + 3 : count, planes + (padded ? 1 : 0), batches);
    ggml_tensor * indices = padded ? ggml_view_3d(values_ctx, index_base, count, planes, batches,
        index_base->nb[1], index_base->nb[2], sizeof(int32_t)) : index_base;
    // ggml_cuda_op_get_rows requires contiguous I32 dimension0. Exercise the
    // supported noncontiguous plane/batch strides without bypassing that guard.
    ggml_backend_buffer_t values = ggml_backend_alloc_ctx_tensors(values_ctx, backend);
    check(values != nullptr, "input allocation failed");
    std::vector<uint8_t> source_bytes(ggml_nbytes(source_base));
    std::vector<uint8_t> index_bytes(ggml_nbytes(index_base), 0);
    for (size_t index = 0; index < source_bytes.size() / sizeof(uint32_t); ++index) {
        const uint32_t value = value_bits(static_cast<int>(index));
        std::memcpy(source_bytes.data() + index * sizeof(value), &value, sizeof(value));
    }
    std::vector<uint32_t> expected(static_cast<size_t>(count) * planes * batches);
    const size_t source_offset = padded ? sizeof(float) : 0;
    const size_t index_offset = padded ? sizeof(int32_t) : 0;
    for (int batch = 0; batch < batches; ++batch)
        for (int plane = 0; plane < planes; ++plane)
            for (int out = 0; out < count; ++out) {
                // Reordering plus duplicate, first and last source indices.
                const int32_t selected = out % 5 == 0 ? rows - 1 : (out * 17 + plane * 3 + batch) % rows;
                const size_t index_position = index_offset + out * indices->nb[0] +
                    plane * indices->nb[1] + batch * indices->nb[2];
                std::memcpy(index_bytes.data() + index_position, &selected, sizeof(selected));
                const size_t position = source_offset + selected * source->nb[1] +
                    plane * source->nb[2] + batch * source->nb[3];
                std::memcpy(&expected[(static_cast<size_t>(batch) * planes + plane) * count + out],
                    source_bytes.data() + position, sizeof(uint32_t));
            }
    ggml_backend_tensor_set(source_base, source_bytes.data(), 0, source_bytes.size());
    ggml_backend_tensor_set(index_base, index_bytes.data(), 0, index_bytes.size());
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 64, false);
    ggml_tensor * output = ggml_get_rows(graph_ctx, source, indices);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    check(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    for (int replay = 0; replay < 3; ++replay) {
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "gather compute failed");
        std::vector<uint32_t> actual(expected.size());
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(uint32_t));
        check(actual == expected, "gather differs from CPU-reference bytes");
    }
    std::cout << "scalar-gather count=" << count << " planes=" << planes
              << " batches=" << batches << " padded=" << padded << " byte_exact=1\n";
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(values);
    ggml_free(graph_ctx);
    ggml_free(values_ctx);
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: qw-scalar-get-rows-probe <isolated ggml-cuda.dll>");
#ifdef _WIN32
        _putenv_s("QW_CUDA_BACKEND", argv[1]);
#else
        setenv("QW_CUDA_BACKEND", argv[1], 1);
#endif
        questwend::initialize_backends();
        ggml_backend_reg_t registry = ggml_backend_reg_by_name("CUDA");
        check(registry != nullptr, "local CUDA registry unavailable");
        ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
        check(backend != nullptr, "local CUDA backend unavailable");
        for (int count : {1, 2, 31, 32, 255, 256, 257, 32768, 131072}) {
            run_case(backend, count, 1, 1, false);
            run_case(backend, count, 3, 2, true);
        }
        ggml_backend_free(backend);
        std::cout << "all18 scalar gather cases passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "scalar gather gate failed: " << error.what() << '\n';
        return 1;
    }
}
