#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "expert_cost_io.h"
#include <cmath>
#include <functional>

namespace expert_cost {
using Phases = std::map<std::string, double>;
inline void measure(Phases & phases, const std::string & name, const std::function<void()> & operation) {
    const auto started = Clock::now();
    operation();
    phases[name] += elapsed(started);
}

class RoutedGraph {
    ggml_backend_t backend_;
    ggml_context * weights_context_ = nullptr, * context_ = nullptr;
    ggml_backend_buffer_t weights_buffer_ = nullptr, compute_buffer_ = nullptr;
    ggml_tensor * weights_[3] = {nullptr, nullptr, nullptr};
    void * original_[3] = {nullptr, nullptr, nullptr};
    ggml_tensor * input_ = nullptr, * ids_ = nullptr, * router_ = nullptr, * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::vector<float> input_data_, router_data_, result_;
    std::vector<int32_t> id_data_;
    int active_, tokens_;
public:
    double allocation_ms = 0;
    RoutedGraph(ggml_backend_t backend, const Expert & shape, int bank, int active, int tokens)
        : backend_(backend), active_(active), tokens_(tokens) {
        const auto started = Clock::now();
        ggml_init_params parameters{16 * 1024 * 1024, nullptr, true};
        weights_context_ = ggml_init(parameters);
        for (int role = 0; role < 3; ++role)
            weights_[role] = ggml_new_tensor_3d(weights_context_, (ggml_type) shape[role].type,
                shape[role].columns, shape[role].rows, bank);
        weights_buffer_ = ggml_backend_alloc_ctx_tensors(weights_context_, backend);
        if (!weights_buffer_) throw std::runtime_error("Cannot allocate calibration weights");
        ggml_backend_buffer_set_usage(weights_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int role = 0; role < 3; ++role) original_[role] = weights_[role]->data;
        context_ = ggml_init(parameters);
        const int64_t embedding = shape[0].columns;
        input_ = ggml_new_tensor_3d(context_, GGML_TYPE_F32, embedding, 1, tokens);
        ids_ = ggml_new_tensor_2d(context_, GGML_TYPE_I32, active, tokens);
        router_ = ggml_new_tensor_2d(context_, GGML_TYPE_F32, active, tokens);
        auto gate = ggml_mul_mat_id(context_, weights_[0], input_, ids_);
        auto up = ggml_mul_mat_id(context_, weights_[1], input_, ids_);
        auto hidden = ggml_swiglu_split(context_, gate, up);
        auto expert_output = ggml_mul_mat_id(context_, weights_[2], hidden, ids_);
        if (tokens == 1) {
            // Match the existing CPU borrowed path and GPU decode reduction.
            auto transposed = ggml_cont(context_, ggml_transpose(context_,
                ggml_reshape_2d(context_, expert_output, embedding, active)));
            output_ = ggml_mul_mat(context_, transposed, router_);
        } else {
            // Match the GPU prefill reduction. Batched CPU execution remains
            // a proposed extension, clearly identified in each result.
            auto scaled = ggml_mul(context_, expert_output, ggml_reshape_3d(context_, router_, 1, active, tokens));
            ggml_tensor * reduced = nullptr;
            for (int index = 0; index < active; ++index) {
                auto view = ggml_view_2d(context_, scaled, embedding, tokens,
                    scaled->nb[2], index * scaled->nb[1]);
                reduced = index ? ggml_add(context_, reduced, view) : view;
            }
            output_ = active == 1 ? ggml_cont(context_, reduced) : reduced;
        }
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(context_, 128, false);
        ggml_build_forward_expand(graph_, output_);
        compute_buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend_);
        if (!compute_buffer_) throw std::runtime_error("Cannot allocate routed calibration graph");
        input_data_.resize(embedding * tokens);
        for (size_t index = 0; index < input_data_.size(); ++index)
            input_data_[index] = ((int) (index % 97) - 48) / 97.0f * 3.4641016f;
        id_data_.resize(active * tokens);
        router_data_.resize(active * tokens);
        result_.resize(embedding * tokens);
        allocation_ms = elapsed(started);
    }
    ~RoutedGraph() {
        for (int role = 0; role < 3; ++role) if (weights_[role]) weights_[role]->data = original_[role];
        if (compute_buffer_) ggml_backend_buffer_free(compute_buffer_);
        if (weights_buffer_) ggml_backend_buffer_free(weights_buffer_);
        if (context_) ggml_free(context_);
        if (weights_context_) ggml_free(weights_context_);
    }
    void borrow(const Expert & shape, uint8_t * data, const std::vector<size_t> & offsets, int index) {
        for (int role = 0; role < 3; ++role)
            weights_[role]->data = data + offsets[role] + index * shape[role].bytes;
    }
    void copy_from(const Expert & shape, const uint8_t * data, const std::vector<size_t> & offsets, int index) {
        for (int role = 0; role < 3; ++role) {
            weights_[role]->data = original_[role];
            ggml_backend_tensor_set(weights_[role], data + offsets[role] + index * shape[role].bytes,
                                    0, shape[role].bytes);
        }
        ggml_backend_synchronize(backend_);
    }
    void upload(const Expert & shape, const uint8_t * data, const std::vector<size_t> & offsets, int count) {
        for (int role = 0; role < 3; ++role)
            ggml_backend_tensor_set_async(backend_, weights_[role], data + offsets[role], 0, count * shape[role].bytes);
        ggml_backend_synchronize(backend_);
    }
    void download(const Expert & shape, uint8_t * data, const std::vector<size_t> & offsets, int count) {
        for (int role = 0; role < 3; ++role)
            ggml_backend_tensor_get_async(backend_, weights_[role], data + offsets[role], 0, count * shape[role].bytes);
        ggml_backend_synchronize(backend_);
    }
    const std::vector<float> & compute(const std::vector<int> & selected,
                                      const std::vector<float> & router, Phases & phases) {
        for (int token = 0; token < tokens_; ++token)
            for (int expert = 0; expert < active_; ++expert) {
                id_data_[token * active_ + expert] = selected[expert];
                router_data_[token * active_ + expert] = router[expert];
            }
        measure(phases, "activation_inputs_ms", [&] {
            ggml_backend_tensor_set(input_, input_data_.data(), 0, input_data_.size() * sizeof(float));
            ggml_backend_tensor_set(ids_, id_data_.data(), 0, id_data_.size() * sizeof(int32_t));
            ggml_backend_tensor_set(router_, router_data_.data(), 0, router_data_.size() * sizeof(float));
            ggml_backend_synchronize(backend_);
        });
        measure(phases, "routed_graph_ms", [&] {
            if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Routed expert compute failed");
            ggml_backend_synchronize(backend_);
        });
        measure(phases, "activation_output_ms", [&] {
            ggml_backend_tensor_get(output_, result_.data(), 0, result_.size() * sizeof(float));
            ggml_backend_synchronize(backend_);
        });
        return result_;
    }
};

// Mirrors Runtime::cpu_experts' borrowed RAM path: one routed graph per
// separately allocated expert, then result accumulation on the CPU.
class BorrowedCpuWork {
    std::unique_ptr<RoutedGraph> graph_;
    std::vector<float> sum_;
    const Expert & shape_;
    bool copy_weights_;
public:
    double allocation_ms = 0;
    BorrowedCpuWork(ggml_backend_t backend, const Expert & shape, int active, int tokens, bool copy_weights = false)
        : shape_(shape), copy_weights_(copy_weights) {
        const auto started = Clock::now();
        graph_.reset(new RoutedGraph(backend, shape, 1, 1, tokens));
        sum_.resize(shape[0].columns * tokens);
        allocation_ms = elapsed(started);
    }
    const std::vector<float> & compute(uint8_t * data, const std::vector<size_t> & offsets,
                                      const std::vector<int> & selected, const std::vector<float> & router,
                                      Phases & phases) {
        std::fill(sum_.begin(), sum_.end(), 0.0f);
        for (size_t index = 0; index < selected.size(); ++index) {
            if (copy_weights_)
                measure(phases, "cpu_weight_copy_ms", [&] { graph_->copy_from(shape_, data, offsets, selected[index]); });
            else graph_->borrow(shape_, data, offsets, selected[index]);
            const auto & part = graph_->compute({0}, {router[index]}, phases);
            measure(phases, "cpu_accumulate_ms", [&] {
                for (size_t column = 0; column < sum_.size(); ++column) sum_[column] += part[column];
            });
        }
        return sum_;
    }
};
}
