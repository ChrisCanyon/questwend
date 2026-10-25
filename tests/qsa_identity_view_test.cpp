// Model-free CPU graph gate. This never initializes a CUDA backend.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static bool overlap(const ggml_tensor * left, const ggml_tensor * right) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(left->data);
    const uintptr_t b = reinterpret_cast<uintptr_t>(right->data);
    return a < b + ggml_nbytes(right) && b < a + ggml_nbytes(left);
}

static std::vector<uint8_t> bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

struct Result {
    std::vector<std::vector<uint8_t>> members, pooled;
    bool view_used = false;
};

static Result run_graph(ggml_backend_t backend, bool requested_view, int ratio,
                        int rows, int dimensions, bool padded, bool special_bits) {
    ggml_init_params init{};
    init.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false);
    init.no_alloc = true;
    ggml_context * state_context = ggml_init(init);
    ggml_context * context = ggml_init(init);
    check(context && state_context, "context allocation failed");
    const int blocks = rows / ratio;
    const int count = blocks * ratio;
    check(blocks > 0, "bad fixture dimensions");
    ggml_tensor * storage = ggml_new_tensor_2d(state_context, GGML_TYPE_F32,
        dimensions + (padded ? 8 : 0), rows);
    ggml_tensor * raw = padded ? ggml_view_2d(state_context, storage, dimensions,
        rows, storage->nb[1], 0) : storage;
    ggml_backend_buffer_t state_buffer = ggml_backend_alloc_ctx_tensors(state_context, backend);
    check(state_buffer != nullptr, "state allocation failed");
    std::vector<float> initial(static_cast<size_t>(storage->ne[0]) * rows);
    for (size_t index = 0; index < initial.size(); ++index) {
        if (special_bits) {
            const uint32_t patterns[] = {0u, 0x80000000u, 1u, 0x7fc01234u,
                0x3f800001u, 0xbf800001u, 0x00800001u};
            const uint32_t value = patterns[index % 7];
            std::memcpy(&initial[index], &value, sizeof(value));
        } else {
            initial[index] = static_cast<float>(static_cast<int>(index % 97) - 48) / 32.0f;
        }
    }
    ggml_backend_tensor_set(storage, initial.data(), 0, initial.size() * sizeof(float));
    std::vector<uint8_t> expected(initial.size() * sizeof(float));
    std::memcpy(expected.data(), initial.data(), expected.size());

    ggml_cgraph * graph = ggml_new_graph_custom(context, 256, false);
    ggml_tensor * update = ggml_new_tensor_2d(context, GGML_TYPE_F32, dimensions, 1);
    ggml_tensor * position = ggml_new_tensor_1d(context, GGML_TYPE_I64, 1);
    ggml_set_input(update);
    ggml_set_input(position);
    // Match runtime's explicit expansion of the side-effecting cache write,
    // while the later source view reads the original persistent cache tensor.
    ggml_build_forward_expand(graph, ggml_set_rows(context, raw, update, position));
    ggml_tensor * indices = ggml_new_tensor_1d(context, GGML_TYPE_I32, count);
    ggml_set_input(indices);
    ggml_set_name(indices, "inp_qsa_blk_cells");
    const bool supported = raw->type == GGML_TYPE_F32 && raw->ne[0] == dimensions &&
        raw->ne[2] == 1 && raw->ne[3] == 1 && raw->nb[0] == sizeof(float) &&
        raw->nb[1] == static_cast<size_t>(dimensions) * sizeof(float) &&
        ggml_is_contiguous(raw) && count > 0 && count <= raw->ne[1];
    Result result;
    result.view_used = requested_view && supported;
    ggml_tensor * members;
    if (result.view_used) {
        ggml_build_forward_expand(graph, indices);
        members = ggml_view_2d(context, raw, dimensions, count, raw->nb[1], 0);
    } else {
        members = ggml_get_rows(context, raw, indices);
    }
    members = ggml_reshape_3d(context, members, dimensions, ratio, blocks);
    ggml_set_output(members);
    ggml_build_forward_expand(graph, members);
    ggml_tensor * pooled = nullptr;
    for (int index = 0; index < ratio; ++index) {
        ggml_tensor * slice = ggml_cont(context, ggml_view_2d(context, members,
            dimensions, blocks, members->nb[2], static_cast<size_t>(index) * members->nb[1]));
        pooled = pooled ? ggml_add(context, pooled, slice) : slice;
    }
    pooled = ggml_scale(context, pooled, 1.0f / static_cast<float>(ratio));
    ggml_set_output(pooled);
    ggml_build_forward_expand(graph, pooled);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    check(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    ggml_tensor * leaf = ggml_graph_get_tensor(graph, "inp_qsa_blk_cells");
    check(leaf == indices && leaf->data != nullptr, "named index leaf disappeared/unallocated");
    check(!overlap(leaf, update) && !overlap(leaf, position), "named leaf aliases another input");
    if (result.view_used) {
        for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            const ggml_tensor * node = ggml_graph_node(graph, index);
            if (node->op != GGML_OP_NONE && !node->view_src)
                check(!overlap(leaf, node), "unused named leaf aliases a graph writer");
        }
    }
    std::vector<int32_t> ids(count);
    for (int index = 0; index < count; ++index) ids[index] = index;
    const int64_t positions[] = {0, ratio - 1, ratio, rows - 1};
    for (int replay = 0; replay < 4; ++replay) {
        std::vector<float> values(dimensions);
        for (int column = 0; column < dimensions; ++column)
            values[column] = static_cast<float>((replay + 1) * 11 - column % 19) / 16.0f;
        const int64_t row = positions[replay];
        ggml_backend_tensor_set(update, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(position, &row, 0, sizeof(row));
        ggml_backend_tensor_set(indices, ids.data(), 0, ids.size() * sizeof(int32_t));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph replay failed");
        std::memcpy(expected.data() + static_cast<size_t>(row) * raw->nb[1],
            values.data(), values.size() * sizeof(float));
        check(bytes(storage) == expected, "raw in-place write ordering changed persistent cache");
        std::vector<uint8_t> prefix(static_cast<size_t>(count) * dimensions * sizeof(float));
        for (int member = 0; member < count; ++member)
            std::memcpy(prefix.data() + static_cast<size_t>(member) * dimensions * sizeof(float),
                expected.data() + static_cast<size_t>(member) * raw->nb[1], dimensions * sizeof(float));
        result.members.push_back(bytes(members));
        check(result.members.back() == prefix, "members differ from authoritative written source bytes");
        if (!special_bits) result.pooled.push_back(bytes(pooled));
    }
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(state_buffer);
    ggml_free(context);
    ggml_free(state_context);
    return result;
}

int main() {
    ggml_backend_t backend = nullptr;
    try {
        backend = ggml_backend_cpu_init();
        check(backend != nullptr, "CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(backend, 1);
        int cases = 0;
        for (int ratio : {1, 2, 4, 8}) {
            for (bool padded : {false, true}) {
                for (bool special : {false, true}) {
                    const int rows = ratio * 5 + (ratio > 1 ? ratio - 1 : 0);
                    const auto reference = run_graph(backend, false, ratio, rows, 128, padded, special);
                    const auto candidate = run_graph(backend, true, ratio, rows, 128, padded, special);
                    check(candidate.view_used == !padded, "guarded layout fallback failed");
                    check(reference.members == candidate.members, "old/view member bytes differ");
                    check(reference.pooled == candidate.pooled, "original left-add/scale output differs");
                    ++cases;
                }
            }
        }
        ggml_backend_free(backend);
        std::cout << "QSA identity view: " << cases << " cases, four writes/replays each, exact bytes\n";
        return 0;
    } catch (const std::exception & error) {
        if (backend) ggml_backend_free(backend);
        std::cerr << "QSA identity view failed: " << error.what() << '\n';
        return 1;
    }
}
