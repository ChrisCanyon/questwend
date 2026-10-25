#pragma once
#include "grouped_cost_graph.h"
#include "real_grouped_client.h"
#include "backend_loader.h"

namespace grouped_gate {

// Numerical shadow only. Reads the SAME four selected real GGUF experts and
// computes their weighted contribution on the primary GPU with native ggml.
// This intentionally adds work and MUST NOT be included in a speed claim.
class NativeReference {
    std::map<std::pair<int, int>, expert_cost::Expert> experts_;
    expert_cost::Expert shape_;
    std::map<int, std::unique_ptr<expert_cost::RoutedGraph>> graphs_;
    expert_cost::ExpertReader reader_{false, 4};
    std::unique_ptr<expert_cost::HostBuffer> staging_;
    ggml_backend_t backend_ = nullptr;
public:
    explicit NativeReference(const Json & manifest) {
        for (const auto & entry : manifest.at("signatures").at(0).at("experts")) {
            expert_cost::Expert expert;
            for (const auto & slab : entry.at("weights"))
                expert.push_back({slab.at("path"), slab.at("offset"), slab.at("bytes"),
                    slab.at("type"), slab.at("shape")[0], slab.at("shape")[1]});
            experts_.emplace(std::make_pair(entry.at("layer").get<int>(), entry.at("expert").get<int>()), std::move(expert));
        }
        if (experts_.empty()) throw std::runtime_error("Empty native reference manifest");
        shape_ = experts_.begin()->second;
        questwend::initialize_backends();
        auto registry = ggml_backend_reg_by_name("CUDA");
        if (!registry) throw std::runtime_error("Native numerical shadow requires CUDA");
        backend_ = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
        if (!backend_) throw std::runtime_error("Native numerical shadow CUDA initialization failed");
        const size_t bytes = 4 * (shape_[0].bytes + shape_[1].bytes + shape_[2].bytes);
        auto pinned = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_));
        if (!pinned) throw std::runtime_error("Native numerical shadow requires pinned staging");
        staging_.reset(new expert_cost::HostBuffer(bytes, pinned));
    }
    ~NativeReference() {
        graphs_.clear(); staging_.reset();
        if (backend_) ggml_backend_free(backend_);
    }
    std::vector<float> execute(const Request & request) {
        const int count = (int) request.experts.size();
        if (count < 1 || count > 4) throw std::runtime_error("Native reference group exceeds bound");
        std::vector<expert_cost::Expert> selected;
        std::vector<int> ids;
        for (int expert : request.experts) {
            auto found = experts_.find({request.layer, expert});
            if (found == experts_.end()) throw std::runtime_error("Native reference real expert missing");
            for (int role = 0; role < 3; ++role)
                if (found->second[role].bytes != shape_[role].bytes || found->second[role].type != shape_[role].type)
                    throw std::runtime_error("Native reference signature mismatch");
            ids.push_back((int) selected.size()); selected.push_back(found->second);
        }
        std::vector<size_t> offsets{0, shape_[0].bytes * count,
            (shape_[0].bytes + shape_[1].bytes) * count};
        reader_.read(selected, ids, offsets, staging_->data);
        auto & graph = graphs_[count];
        if (!graph) graph.reset(new expert_cost::RoutedGraph(backend_, shape_, count, count, 1));
        graph->upload(shape_, staging_->data, offsets, count);
        graph->verify_upload(shape_, staging_->data, offsets, count);
        graph->set_input(request.input);
        expert_cost::Phases ignored;
        return graph->compute(ids, request.weights, ignored);
    }
};
} // namespace grouped_gate
