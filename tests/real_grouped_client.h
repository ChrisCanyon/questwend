#pragma once
#include "runtime.h"
#include "httplib.h"
#include "json.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace grouped_gate {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Request = questwend::Runtime::GroupedExpertRequest;

inline double elapsed_ms(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

inline std::string encode(const Json & metadata, const std::vector<float> & values) {
    const std::string header = metadata.dump();
    if (header.size() > 16384 || values.size() > 2560) throw std::runtime_error("Grouped frame exceeds bound");
    const uint32_t size = static_cast<uint32_t>(header.size());
    std::string output(4, '\0');
    for (int byte = 0; byte < 4; ++byte) output[byte] = static_cast<char>(size >> (24 - 8 * byte));
    output += header;
    output.append(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    return output;
}

inline std::pair<Json, std::vector<float>> decode(const std::string & body) {
    if (body.size() < 4 || body.size() > (1 << 20)) throw std::runtime_error("Invalid grouped reply size");
    const auto * first = reinterpret_cast<const unsigned char *>(body.data());
    const uint32_t bytes = (uint32_t(first[0]) << 24) | (uint32_t(first[1]) << 16) |
        (uint32_t(first[2]) << 8) | first[3];
    if (bytes > 16384 || 4 + size_t(bytes) + 2560 * sizeof(float) != body.size())
        throw std::runtime_error("Invalid grouped reply shape");
    Json metadata = Json::parse(body.substr(4, bytes));
    std::vector<float> values(2560);
    std::memcpy(values.data(), body.data() + 4 + bytes, values.size() * sizeof(float));
    for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite grouped reply");
    return {std::move(metadata), std::move(values)};
}

inline Json compare(const std::vector<float> & reference, const std::vector<float> & actual) {
    if (reference.size() != actual.size()) throw std::runtime_error("Numerical comparison shape mismatch");
    double squared = 0, magnitude = 0, maximum = 0;
    for (size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(reference[index]) || !std::isfinite(actual[index]))
            throw std::runtime_error("Nonfinite numerical comparison");
        const double difference = double(reference[index]) - actual[index];
        squared += difference * difference; magnitude += double(reference[index]) * reference[index];
        maximum = std::max(maximum, std::abs(difference));
    }
    return {{"relative_l2", std::sqrt(squared / std::max(1e-30, magnitude))}, {"maximum_absolute", maximum}};
}

class RemoteClient {
    httplib::Client client_;
    Json identity_;
    std::string fingerprint_, key_;
    uint64_t sequence_ = 0;
public:
    Json calls = Json::array();
    RemoteClient(int port, std::string key, std::string fingerprint)
        : client_("127.0.0.1", port), fingerprint_(std::move(fingerprint)), key_(std::move(key)) {
        client_.set_keep_alive(true);
        client_.set_tcp_nodelay(true);
        client_.set_connection_timeout(8);
        client_.set_read_timeout(15);
        client_.set_write_timeout(15);
        auto reply = client_.Get("/status", {{"X-Experiment-Key", key_}});
        if (!reply || reply->status != 200) throw std::runtime_error("Real grouped worker unavailable");
        identity_ = Json::parse(reply->body);
        if (identity_.at("model_fingerprint") != fingerprint_ || !identity_.value("real_layer_ids", false) ||
            identity_.at("mode") != "ram_cpu" || identity_.at("embedding") != 2560 ||
            identity_.at("signature") != 0 || identity_.at("experts_per_layer") != 512)
            throw std::runtime_error("Real grouped worker identity mismatch");
    }
    Json identity() const { return identity_; }
    std::vector<float> execute(const Request & request) {
        if (request.input.size() != 2560 || request.experts.empty() || request.experts.size() > 4 ||
            request.experts.size() != request.weights.size() || request.position < 0)
            throw std::runtime_error("Real grouped request exceeds strict gate bounds");
        bool known = false;
        for (const auto & value : identity_.at("layers")) known |= value.get<int>() == request.layer;
        if (!known) throw std::runtime_error("Requested layer not resident on worker");
        for (int expert : request.experts) if (expert < 0 || expert >= 512) throw std::runtime_error("Invalid real expert ID");
        Json metadata{{"sequence", ++sequence_}, {"model_fingerprint", fingerprint_}, {"signature", 0},
            {"layer", request.layer}, {"position", request.position}, {"tokens", 1},
            {"experts", request.experts}, {"weights", request.weights}};
        const auto started = Clock::now();
        // Bundled httplib otherwise adds a separate100-Continue exchange for
        // bodies >=1KiB. This bounded authenticated loopback/tunnel request
        // should send its metadata+activation payload without that roundtrip.
        auto reply = client_.Post("/experts", {{"X-Experiment-Key", key_}, {"Expect", ""}},
            encode(metadata, request.input), "application/octet-stream");
        if (!reply || reply->status != 200) throw std::runtime_error("Grouped request failed; reset required");
        auto decoded = decode(reply->body);
        const double wall = elapsed_ms(started);
        const auto & response = decoded.first;
        if (response.at("sequence") != sequence_ || response.at("identity") != identity_ ||
            response.at("tokens") != 1 || response.at("layer") != request.layer ||
            response.at("position") != request.position || response.at("experts") != request.experts ||
            response.at("weight_network_payload_bytes") != 0)
            throw std::runtime_error("Grouped response identity/sequence mismatch");
        calls.push_back({{"layer", request.layer}, {"position", request.position},
            {"experts", request.experts}, {"weights", request.weights}, {"wall_ms", wall},
            {"actual_selected_experts", request.selected_experts}, {"actual_selected_weights", request.selected_weights},
            {"locally_gpu_resident_before_ensure", request.locally_gpu_resident},
            {"locally_ram_resident_before_ensure", request.locally_ram_resident},
            {"worker_phases", response.at("phases")}});
        return std::move(decoded.second);
    }
};
} // namespace grouped_gate
