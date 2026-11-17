#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const auto begin_a = reinterpret_cast<uintptr_t>(a->data);
    const auto begin_b = reinterpret_cast<uintptr_t>(b->data);
    return begin_a < begin_b + ggml_nbytes(b) && begin_b < begin_a + ggml_nbytes(a);
}

// Same important lifetime pattern as QSA: an I32 INPUT is consumed by GET_ROWS,
// then later F32 temporaries are allocated. INPUT does not mean retained.
static void run_allocator_case(bool retained) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    check(context != nullptr, "context allocation failed");
    ggml_backend_t backend = ggml_backend_cpu_init();
    check(backend != nullptr, "CPU backend allocation failed");
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_cgraph * graph = ggml_new_graph_custom(context, 64, false);
    ggml_tensor * source = ggml_new_tensor_2d(context, GGML_TYPE_F32, 16, 64);
    ggml_tensor * indices = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_set_input(source);
    ggml_set_output(source); // Stable reference values across graph replays.
    ggml_set_input(indices);
    if (retained) ggml_set_output(indices);
    ggml_tensor * gathered = ggml_get_rows(context, source, indices);
    ggml_tensor * result = ggml_sum_rows(context, gathered);
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    check(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");

    bool write_alias = false;
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        const ggml_tensor * node = ggml_graph_node(graph, index);
        if (node != indices && node->op != GGML_OP_NONE && !node->view_src)
            write_alias = write_alias || overlaps(indices, node);
    }
    check(retained ? !write_alias : write_alias,
          retained ? "retained input aliases a writer" : "control failed to exercise input alias reuse");

    std::vector<float> values(16 * 64);
    std::vector<int32_t> ids(64);
    for (int row = 0; row < 64; ++row) {
        ids[row] = 63 - row;
        for (int column = 0; column < 16; ++column)
            values[row * 16 + column] = static_cast<float>(row + column);
    }
    ggml_backend_tensor_set(source, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(indices, ids.data(), 0, ids.size() * sizeof(int32_t));
    for (int repeat = 0; repeat < (retained ? 8 : 1); ++repeat) {
        // The unprotected control executes only once: its indices are no longer
        // valid after compute. Retained inputs must survive all eight replays.
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        std::vector<float> actual(64);
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        for (int row = 0; row < 64; ++row)
            check(actual[row] == static_cast<float>(16 * ids[row] + 120), "output differs");
        std::vector<int32_t> after(64);
        ggml_backend_tensor_get(indices, after.data(), 0, after.size() * sizeof(int32_t));
        const bool unchanged = std::memcmp(after.data(), ids.data(), ids.size() * sizeof(int32_t)) == 0;
        check(retained ? unchanged : !unchanged,
              retained ? "retained input bytes were overwritten" : "control did not overwrite freed input bytes");
    }
    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(context);
}

int main() {
    try {
        run_allocator_case(false);
        // Fresh graph/allocator each time also exercises graph replacement.
        for (int rebuild = 0; rebuild < 3; ++rebuild) run_allocator_case(true);
        std::cout << "QSA constant storage tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "QSA constant storage test failed: " << error.what() << '\n';
        return 1;
    }
}
