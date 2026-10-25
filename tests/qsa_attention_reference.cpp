// Bounded diagnostic only: no model weights, routing, state, or service changes.
// Build/run only in a parent-approved isolated GPU window.
#include "backend_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using Json = nlohmann::json;

struct GraphStorage {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~GraphStorage() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
};

static Json difference(const std::vector<float> & actual, const std::vector<double> & reference) {
    double squared = 0, reference_squared = 0, max_absolute = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) throw std::runtime_error("Nonfinite attention output");
        const double error = actual[index] - reference[index];
        squared += error * error;
        reference_squared += reference[index] * reference[index];
        max_absolute = std::max(max_absolute, std::abs(error));
    }
    return {{"relative_l2", std::sqrt(squared / std::max(reference_squared, 1e-300))},
            {"max_absolute", max_absolute}, {"rmse", std::sqrt(squared / actual.size())}};
}

static Json run_case(ggml_backend_t backend, int cells, int finite_count) {
    constexpr int dimension = 512, heads = 8;
    const int valid_cells = cells - 3; // Future/padded rows must never be selected.
    if (finite_count < 1 || finite_count > valid_cells) throw std::invalid_argument("Invalid finite count");
    // Initialize ggml's conversion tables before the CPU reference reads F16.
    GraphStorage storage;
    storage.context = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!storage.context) throw std::runtime_error("Cannot allocate graph metadata");
    std::vector<float> query(dimension * heads);
    std::vector<ggml_fp16_t> keys(size_t(dimension) * cells), values(keys.size());
    for (size_t index = 0; index < query.size(); ++index)
        query[index] = float(int((index * 13 + 7) % 101) - 50) / 101.0f;
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = ggml_fp32_to_fp16(float(int((index * 17 + index / dimension) % 127) - 63) / 64.0f);
        values[index] = ggml_fp32_to_fp16(float(int((index * 23 + 11) % 113) - 56) / 57.0f);
    }
    std::vector<bool> selected(cells, false);
    // Include an incomplete causal tail, then scattered historical rows.
    int selected_count = 0;
    for (int row = valid_cells - 3; row < valid_cells && selected_count < finite_count; ++row) {
        selected[row] = true;
        ++selected_count;
    }
    for (int row = 0; selected_count < finite_count; row = (row + 37) % valid_cells) {
        if (!selected[row]) { selected[row] = true; ++selected_count; }
    }
    std::vector<int> ascending;
    std::vector<ggml_fp16_t> mask(cells, ggml_fp32_to_fp16(-INFINITY));
    for (int row = 0; row < cells; ++row) if (selected[row]) {
        mask[row] = ggml_fp32_to_fp16(0.0f);
        ascending.push_back(row);
    }

    std::vector<double> reference(dimension * heads);
    const float scale = 1.0f / std::sqrt(float(dimension));
    for (int head = 0; head < heads; ++head) {
        std::vector<double> scores(ascending.size());
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t item = 0; item < ascending.size(); ++item) {
            const size_t base = size_t(ascending[item]) * dimension;
            double dot = 0;
            for (int column = 0; column < dimension; ++column)
                dot += double(query[head * dimension + column]) * ggml_fp16_to_fp32(keys[base + column]);
            scores[item] = dot * double(scale);
            maximum = std::max(maximum, scores[item]);
        }
        double denominator = 0;
        for (double & score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (size_t item = 0; item < ascending.size(); ++item) {
            const size_t base = size_t(ascending[item]) * dimension;
            const double weight = scores[item] / denominator;
            for (int column = 0; column < dimension; ++column)
                reference[head * dimension + column] += weight * ggml_fp16_to_fp32(values[base + column]);
        }
    }

    auto * q = ggml_new_tensor_3d(storage.context, GGML_TYPE_F32, dimension, 1, heads);
    auto * k = ggml_new_tensor_3d(storage.context, GGML_TYPE_F16, dimension, cells, 1);
    auto * v = ggml_new_tensor_3d(storage.context, GGML_TYPE_F16, dimension, cells, 1);
    auto * m = ggml_new_tensor_2d(storage.context, GGML_TYPE_F16, cells, 1);
    auto * dense = ggml_flash_attn_ext(storage.context, q, k, v, m, scale, 0.0f, 0.0f);
    auto * sparse = ggml_flash_attn_ext(storage.context, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(sparse, finite_count);
    auto * graph = ggml_new_graph_custom(storage.context, 32, false);
    ggml_build_forward_expand(graph, dense);
    ggml_build_forward_expand(graph, sparse);
    storage.buffer = ggml_backend_alloc_ctx_tensors(storage.context, backend);
    if (!storage.buffer) throw std::runtime_error("Cannot allocate attention fixture");
    ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Attention fixture computation failed");
    ggml_backend_synchronize(backend);
    std::vector<float> dense_output(reference.size()), sparse_output(reference.size());
    ggml_backend_tensor_get(dense, dense_output.data(), 0, dense_output.size() * sizeof(float));
    ggml_backend_tensor_get(sparse, sparse_output.data(), 0, sparse_output.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Repeated attention fixture failed");
    ggml_backend_synchronize(backend);
    std::vector<float> dense_repeat(reference.size()), sparse_repeat(reference.size());
    ggml_backend_tensor_get(dense, dense_repeat.data(), 0, dense_repeat.size() * sizeof(float));
    ggml_backend_tensor_get(sparse, sparse_repeat.data(), 0, sparse_repeat.size() * sizeof(float));
    std::vector<double> dense_reference(dense_output.begin(), dense_output.end());
    return {{"cells", cells}, {"valid_cells", valid_cells}, {"finite_count", finite_count},
        {"finite_ids_ascending", ascending}, {"dimension", dimension}, {"query_heads", heads},
        {"kv_heads", 1}, {"scale", scale},
        {"dense_repeat_exact", std::memcmp(dense_output.data(), dense_repeat.data(), dense_output.size() * sizeof(float)) == 0},
        {"sparse_repeat_exact", std::memcmp(sparse_output.data(), sparse_repeat.data(), sparse_output.size() * sizeof(float)) == 0},
        {"dense_vs_double", difference(dense_output, reference)},
        {"sparse_vs_double", difference(sparse_output, reference)},
        {"sparse_vs_dense", difference(sparse_output, dense_reference)},
        {"requested_sparse_bound", finite_count}, {"actual_sparse_dispatch_verified", false}};
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: qw-qsa-attention-reference output.json");
        questwend::initialize_backends();
        auto registry = ggml_backend_reg_by_name("CUDA");
        if (!registry || ggml_backend_reg_dev_count(registry) < 1)
            throw std::runtime_error("An explicit isolated CUDA backend is required");
        auto backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
        if (!backend) throw std::runtime_error("Cannot initialize CUDA0");
        auto cleanup = std::unique_ptr<void, void (*)(void *)>(backend,
            [](void * value) { ggml_backend_free(static_cast<ggml_backend_t>(value)); });
        Json cases = Json::array();
        for (int count : {2048, 2049, 2050, 2051}) cases.push_back(run_case(backend, 8192, count));
        // 4096 < 2*2051: the sparse heuristic must retain its safe dense fallback.
        cases.push_back(run_case(backend, 4096, 2051));
        Json report{{"state", "completed_numerical_diagnostic_review_required"},
            {"backend", ggml_backend_name(backend)}, {"no_model_weights", true},
            {"cases", cases}, {"quality_gate_passed", false},
            {"note", "Compare errors to CPU double reference and verify sparse kernels in profiler. Different floating reductions are not automatically accepted or called corruption."}};
        std::ofstream output(argv[1]);
        if (!output) throw std::runtime_error("Cannot create diagnostic output");
        output << report.dump(2) << '\n';
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
