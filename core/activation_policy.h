// MoE-Infinity-derived policy, not its PyTorch inference runtime.
// Paper arXiv:2401.14361, Algorithms 1/2 and sections 4-6.
// Reference implementation: EfficientMoE/MoE-Infinity commit
// 7ff99b101b995ec087fde7fccb20b56bdc9b5a64, Apache License 2.0.
// Retain third_party/moe-infinity/LICENSE when distributing this adaptation.
// Differences: bounded CPU arrays, explicit observed-row cosine matching,
// complete GGUF expert handles, top-two one-layer lookahead, and conservative
// transient admission. CPU execution selection remains a separate policy.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace questwend {

class ActivationPolicy {
public:
    ActivationPolicy(int layers, int experts)
        : layers_(layers), experts_(experts), counts_((size_t) layers * experts, 0),
          row_totals_(layers, 0), priorities_(counts_.size(), 0) {
        if (layers <= 0 || experts <= 0) throw std::invalid_argument("Invalid activation policy dimensions");
    }

    void observe(int layer, const int32_t * ids, int count) {
        if (layer < 0 || layer >= layers_ || count < 0) throw std::out_of_range("Invalid activation trace row");
        for (int index = 0; index < count; ++index) {
            if (ids[index] < 0 || ids[index] >= experts_) throw std::out_of_range("Invalid activation expert ID");
            ++counts_[(size_t) layer * experts_ + ids[index]];
            ++row_totals_[layer];
        }
        const double topology = 1.0 - (double) layer / layers_;
        for (int expert = 0; expert < experts_; ++expert) {
            const size_t key = (size_t) layer * experts_ + expert;
            priorities_[key] = (row_totals_[layer] ? (double) counts_[key] / row_totals_[layer] : 0) * topology;
        }
    }

    double priority(size_t logical_key) const { return priorities_.at(logical_key); }

    void save_trace(const std::string & path) const {
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Cannot save activation trace");
        output << "QWACTIVATION 1 " << layers_ << ' ' << experts_ << '\n';
        for (auto count : counts_) output << count << '\n';
    }

    void load_references(const std::string & manifest) {
        std::ifstream listing(manifest);
        if (!listing) throw std::runtime_error("Cannot open activation reference manifest");
        std::string path;
        while (std::getline(listing, path)) {
            if (!path.empty() && path.back() == '\r') path.pop_back();
            if (path.empty()) continue;
            if (references_.size() >= 16) throw std::runtime_error("Activation reference cap exceeded");
            std::ifstream input(path);
            std::string magic;
            int version, layers, experts;
            if (!(input >> magic >> version >> layers >> experts) || magic != "QWACTIVATION" ||
                version != 1 || layers != layers_ || experts != experts_)
                throw std::runtime_error("Activation reference identity/dimensions do not match");
            std::vector<double> trace(counts_.size());
            for (auto & count : trace)
                if (!(input >> count) || !std::isfinite(count) || count < 0)
                    throw std::runtime_error("Invalid activation reference count");
            references_.push_back(std::move(trace));
        }
    }

    // Compare only observed rows. Upstream's literal masking of the stored
    // prefix is not used: an identical observed prefix must match itself.
    int closest_reference(int current_layer) const {
        if (current_layer < 0 || current_layer >= layers_) throw std::out_of_range("Invalid current layer");
        int best = -1;
        double best_score = -1;
        for (size_t reference = 0; reference < references_.size(); ++reference) {
            double similarity = 0;
            int observed = 0;
            for (int layer = 0; layer <= current_layer; ++layer) {
                if (!row_totals_[layer]) continue;
                double dot = 0, current_norm = 0, reference_norm = 0;
                for (int expert = 0; expert < experts_; ++expert) {
                    const size_t key = (size_t) layer * experts_ + expert;
                    const double current = (double) counts_[key];
                    const double saved = references_[reference][key];
                    dot += current * saved;
                    current_norm += current * current;
                    reference_norm += saved * saved;
                }
                if (current_norm && reference_norm) similarity += dot / std::sqrt(current_norm * reference_norm);
                ++observed;
            }
            const double score = observed ? similarity / observed : 0;
            if (score > best_score) { best = (int) reference; best_score = score; }
        }
        return best;
    }

    std::vector<int> predict_next(int current_layer, size_t maximum) const {
        const int future_layer = current_layer + 1;
        if (future_layer >= layers_) return {};
        const int reference = closest_reference(current_layer);
        if (reference < 0) return {};
        std::vector<int> result;
        for (int expert = 0; expert < experts_; ++expert)
            if (references_[reference][(size_t) future_layer * experts_ + expert] > 0) result.push_back(expert);
        std::stable_sort(result.begin(), result.end(), [&](int left, int right) {
            return references_[reference][(size_t) future_layer * experts_ + left] >
                   references_[reference][(size_t) future_layer * experts_ + right];
        });
        if (result.size() > maximum) result.resize(maximum);
        // MoE-Infinity's linear distance factor is the same positive constant
        // for every candidate in this bounded one-layer lookahead.
        return result;
    }

private:
    int layers_, experts_;
    std::vector<uint64_t> counts_, row_totals_;
    std::vector<double> priorities_;
    std::vector<std::vector<double>> references_;
};

} // namespace questwend
