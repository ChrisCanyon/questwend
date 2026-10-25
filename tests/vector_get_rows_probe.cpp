// Exact indexed-copy reference for width128 packed F32 rows. Separate processes
// set QWEN_CUDA_VECTOR_GET_ROWS=0/1, because the runtime reads the flag once.
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

static uint32_t value_bits(size_t index) {
    const uint32_t special[] = {0, 0x80000000u, 0x3f800000u, 0xbf800000u,
        0x7fc01234u, 0x7f800000u, 0xff800000u, 1, 0x7fa01234u};
    return index % 3 ? 0x3f000000u + static_cast<uint32_t>(index) : special[(index / 3) % 9];
}

static void run_case(ggml_backend_t backend, int width, int count,
                     int heads, int batches, bool padded, bool unaligned) {
    const int rows = 67;
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
    params.no_alloc = true;
    auto * input_ctx = ggml_init(params);
    auto * graph_ctx = ggml_init(params);
    check(input_ctx && graph_ctx, "context allocation failed");
    const size_t source_offset = unaligned ? sizeof(float) : padded ? 16 : 0;
    auto * source_base = ggml_new_tensor_4d(input_ctx, GGML_TYPE_F32,
        width + (padded || unaligned ? 4 : 0), rows, heads, batches);
    auto * source = source_offset ? ggml_view_4d(input_ctx, source_base, width, rows, heads, batches,
        source_base->nb[1], source_base->nb[2], source_base->nb[3], source_offset) : source_base;
    auto * index_base = ggml_new_tensor_3d(input_ctx, GGML_TYPE_I32,
        count + (padded ? 3 : 0), heads + (padded ? 1 : 0), batches);
    const size_t index_offset = padded ? sizeof(int32_t) : 0;
    auto * indices = padded ? ggml_view_3d(input_ctx, index_base, count, heads, batches,
        index_base->nb[1], index_base->nb[2], index_offset) : index_base;
    // Dimension0 must remain contiguous: the public CUDA GET_ROWS operator
    // asserts nb0==sizeof(I32). Higher-dimensional strides may be padded.
    auto * inputs = ggml_backend_alloc_ctx_tensors(input_ctx, backend);
    check(inputs != nullptr, "input tensor allocation failed");
    std::vector<uint8_t> source_bytes(ggml_nbytes(source_base));
    std::vector<uint8_t> index_bytes(ggml_nbytes(index_base), 0);
    for (size_t i = 0; i < source_bytes.size() / 4; ++i) {
        const uint32_t bits = value_bits(i);
        std::memcpy(source_bytes.data() + 4*i, &bits, 4);
    }
    auto * graph = ggml_new_graph_custom(graph_ctx, 64, false);
    auto * output = ggml_get_rows(graph_ctx, source, indices);
    if (padded) {
        output->nb[1] = static_cast<size_t>(width + 4) * sizeof(float);
        output->nb[2] = output->nb[1] * (count + 3);
        output->nb[3] = output->nb[2] * (heads + 2);
    }
    std::vector<uint8_t> expected(ggml_nbytes(output), 0);
    for (int batch = 0; batch < batches; ++batch)
        for (int head = 0; head < heads; ++head)
            for (int row = 0; row < count; ++row) {
                const int32_t selected = row % 5 == 0 ? rows - 1 : (row * 17 + head * 3 + batch) % rows;
                const size_t idx = index_offset + row*indices->nb[0] + head*indices->nb[1] + batch*indices->nb[2];
                std::memcpy(index_bytes.data() + idx, &selected, 4);
                const size_t src = source_offset + selected*source->nb[1] + head*source->nb[2] + batch*source->nb[3];
                const size_t dst = row*output->nb[1] + head*output->nb[2] + batch*output->nb[3];
                std::memcpy(expected.data() + dst, source_bytes.data() + src, width*4);
            }
    ggml_backend_tensor_set(source_base, source_bytes.data(), 0, source_bytes.size());
    ggml_backend_tensor_set(index_base, index_bytes.data(), 0, index_bytes.size());
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    check(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    for (int replay = 0; replay < 3; ++replay) {
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA GET_ROWS failed");
        std::vector<uint8_t> actual(expected.size());
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size());
        for (int batch = 0; batch < batches; ++batch)
            for (int head = 0; head < heads; ++head)
                for (int row = 0; row < count; ++row) {
                    const size_t offset = row*output->nb[1] + head*output->nb[2] + batch*output->nb[3];
                    check(std::memcmp(actual.data() + offset, expected.data() + offset, width*4) == 0,
                        "GET_ROWS differs from authoritative CPU copy bits");
                }
    }
    std::cout << "vector-gather width=" << width << " rows=" << count << " heads=" << heads
              << " batches=" << batches << " padded=" << padded << " unaligned=" << unaligned
              << " byte_exact=1\n";
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(inputs);
    ggml_free(graph_ctx);
    ggml_free(input_ctx);
}

int main(int argc, char ** argv) {
    try {
        check(argc == 2, "Usage: qw-vector-get-rows-probe <isolated ggml-cuda.dll>");
#ifdef _WIN32
        _putenv_s("QW_CUDA_BACKEND", argv[1]);
#else
        setenv("QW_CUDA_BACKEND", argv[1], 1);
#endif
        questwend::initialize_backends();
        auto registry = ggml_backend_reg_by_name("CUDA");
        check(registry != nullptr, "CUDA registry unavailable");
        auto backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
        check(backend != nullptr, "CUDA initialization failed");
        for (int rows : {1, 7, 8, 9, 127, 128, 129, 32768}) {
            run_case(backend, 128, rows, 1, 1, false, false);
            run_case(backend, 128, rows, 3, 2, true, false);
        }
        run_case(backend, 128, 129, 3, 2, true, true); // Alignment fallback.
        run_case(backend, 124, 129, 3, 2, true, false); // Width fallback.
        run_case(backend, 132, 129, 3, 2, true, false); // Width fallback.
        ggml_backend_free(backend);
        std::cout << "all19 vector GET_ROWS cases passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vector GET_ROWS gate failed: " << error.what() << '\n';
        return 1;
    }
}
