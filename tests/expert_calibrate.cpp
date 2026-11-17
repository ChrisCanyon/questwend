// Calibrate one complete, unmodified expert on local or replica-backed devices.
// Remote requests load weights by identifier; only activations/results travel.
#include "backend_loader.h"
#include "rpc_weights.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Json distribution(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return {{"samples", values.size()}, {"min_ms", values.front()},
            {"median_ms", values[values.size() / 2]},
            {"p95_ms", values[std::min(values.size() - 1, (size_t) std::ceil(values.size() * .95) - 1)]},
            {"mean_ms", std::accumulate(values.begin(), values.end(), 0.0) / values.size()}};
}

int main(int argc, char ** argv) {
    try {
        if (argc != 8) throw std::runtime_error(
            "Usage: qw-expert-calibrate manifest.json cpu|gpu|rpc batch repeats endpoint device output-prefix");
        std::ifstream manifest_stream(argv[1]);
        Json manifest;
        manifest_stream >> manifest;
        const std::string mode = argv[2], endpoint = argv[5];
        const int batch = std::stoi(argv[3]), repeats = std::stoi(argv[4]), device = std::stoi(argv[6]);
        if (batch < 1 || batch > 1024 || repeats < 3 || repeats > 1000)
            throw std::runtime_error("Invalid batch or repeat count");
        const bool remote = mode == "rpc";
        const bool float_reference = mode == "cpu-f32";
        questwend::initialize_backends();
        ggml_backend_t backend = nullptr;
        if (mode == "cpu" || float_reference) {
            backend = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(backend, manifest.value("cpu_threads", 8));
        } else if (mode == "gpu") {
            auto registry = ggml_backend_reg_by_name("CUDA");
            if (!registry || device < 0 || (size_t) device >= ggml_backend_reg_dev_count(registry))
                throw std::runtime_error("Requested CUDA device unavailable");
            backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, device), nullptr);
        } else if (remote && endpoint.rfind("127.0.0.1:", 0) == 0) {
            auto registry = ggml_backend_reg_by_name("RPC");
            if (!registry) throw std::runtime_error("RPC backend unavailable");
            using Add = ggml_backend_reg_t (*)(const char *);
            auto add = reinterpret_cast<Add>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_rpc_add_server"));
            auto node = add ? add(endpoint.c_str()) : nullptr;
            if (!node || device < 0 || (size_t) device >= ggml_backend_reg_dev_count(node))
                throw std::runtime_error("Requested remote device unavailable");
            backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(node, device), nullptr);
        } else throw std::runtime_error("Unknown mode or non-loopback RPC endpoint");
        if (!backend) throw std::runtime_error("Backend initialization failed");

        ggml_init_params parameters{16 * 1024 * 1024, nullptr, true};
        auto weight_context = ggml_init(parameters);
        std::vector<ggml_tensor *> weights;
        std::vector<std::vector<uint8_t>> payloads;
        std::vector<uint64_t> hashes;
        std::vector<size_t> offsets, sizes;
        uint64_t weight_bytes = 0;
        double source_read_ms = 0;
        for (const auto & item : manifest.at("weights")) {
            const auto stored_type = (ggml_type) item.at("type").get<int>();
            auto weight = ggml_new_tensor_2d(weight_context, float_reference ? GGML_TYPE_F32 : stored_type,
                item.at("shape")[0].get<int64_t>(), item.at("shape")[1].get<int64_t>());
            const auto name = item.at("role").get<std::string>();
            ggml_set_name(weight, name.c_str());
            const size_t source_bytes = item.at("bytes").get<size_t>();
            const size_t bytes = ggml_nbytes(weight);
            if (!float_reference && bytes != source_bytes) throw std::runtime_error("Manifest tensor byte count differs");
            weights.push_back(weight);
            hashes.push_back(std::stoull(item.at("hash").get<std::string>(), nullptr, 16));
            offsets.push_back(0);
            sizes.push_back(bytes);
            weight_bytes += bytes;
            if (!remote) {
                const auto started = Clock::now();
                std::ifstream source(item.at("path").get<std::string>(), std::ios::binary);
                source.seekg(item.at("offset").get<uint64_t>());
                std::vector<uint8_t> stored(source_bytes);
                source.read((char *) stored.data(), source_bytes);
                if ((size_t) source.gcount() != source_bytes) throw std::runtime_error("Short expert file read");
                if (float_reference) {
                    payloads.emplace_back(bytes);
                    ggml_get_type_traits(stored_type)->to_float(stored.data(),
                        (float *) payloads.back().data(), ggml_nelements(weight));
                } else payloads.push_back(std::move(stored));
                source_read_ms += milliseconds(started);
            }
        }
        if (weights.size() != 3) throw std::runtime_error("A complete gate/up/down expert is required");
        auto weight_buffer = ggml_backend_alloc_ctx_tensors(weight_context, backend);
        if (!weight_buffer) throw std::runtime_error("Cannot allocate expert weights");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<double> load_times, readback_times;
        for (int iteration = 0; iteration < repeats; ++iteration) {
            const auto started = Clock::now();
            if (remote) questwend::fill_remote_experts(weights, hashes, offsets, sizes);
            else for (size_t role = 0; role < weights.size(); ++role)
                ggml_backend_tensor_set(weights[role], payloads[role].data(), 0, sizes[role]);
            ggml_backend_synchronize(backend);
            load_times.push_back(milliseconds(started));
        }
        // Local D2H/RAM readback calibration only. Never stream remote weights.
        if (!remote) for (int iteration = 0; iteration < repeats; ++iteration) {
            const auto started = Clock::now();
            for (size_t role = 0; role < weights.size(); ++role)
                ggml_backend_tensor_get(weights[role], payloads[role].data(), 0, sizes[role]);
            readback_times.push_back(milliseconds(started));
        }

        auto context = ggml_init(parameters);
        auto input = ggml_new_tensor_2d(context, GGML_TYPE_F32, weights[0]->ne[0], batch);
        auto gate = ggml_mul_mat(context, weights[0], input);
        auto up = ggml_mul_mat(context, weights[1], input);
        auto product = ggml_mul(context, ggml_silu(context, gate), up);
        auto output = ggml_mul_mat(context, weights[2], product);
        auto graph = ggml_new_graph_custom(context, 128, false);
        ggml_build_forward_expand(graph, output);
        auto compute_buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!compute_buffer) throw std::runtime_error("Cannot allocate expert activations");
        std::vector<float> inputs(ggml_nelements(input)), outputs(ggml_nelements(output));
        for (size_t index = 0; index < inputs.size(); ++index)
            inputs[index] = (float((int) (index % 97) - 48) / 97.0f) * 3.4641016f;
        std::vector<double> upload_times, compute_times, result_times;
        for (int iteration = -2; iteration < repeats; ++iteration) {
            auto started = Clock::now();
            ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
            ggml_backend_synchronize(backend);
            const double upload = milliseconds(started);
            started = Clock::now();
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Expert computation failed");
            ggml_backend_synchronize(backend);
            const double compute = milliseconds(started);
            started = Clock::now();
            ggml_backend_tensor_get(output, outputs.data(), 0, outputs.size() * sizeof(float));
            const double result = milliseconds(started);
            if (iteration >= 0) {
                upload_times.push_back(upload); compute_times.push_back(compute); result_times.push_back(result);
            }
        }
        double square_sum = 0;
        for (float value : outputs) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite expert result");
            square_sum += (double) value * value;
        }
        const std::string prefix = argv[7];
        std::ofstream binary(prefix + ".output.f32", std::ios::binary);
        binary.write((const char *) outputs.data(), outputs.size() * sizeof(float));
        Json report{{"state", "done"}, {"backend", ggml_backend_name(backend)}, {"mode", mode},
            {"batch", batch}, {"weight_bytes", weight_bytes}, {"source_buffered_read_ms", source_read_ms},
            {"weight_load", distribution(load_times)}, {"activation_upload", distribution(upload_times)},
            {"expert_compute", distribution(compute_times)}, {"activation_result", distribution(result_times)},
            {"input_bytes", inputs.size() * sizeof(float)}, {"output_bytes", outputs.size() * sizeof(float)},
            {"output_l2", std::sqrt(square_sum)}, {"remote_weight_payload_bytes", 0},
            {"float_reference", float_reference}, {"input_distribution", "deterministic approximately unit RMS"},
            {"remote_compute_timing_includes_rpc", remote}, {"replica_load_is_cold_disk_measurement", false}};
        if (!remote) report["weight_readback"] = distribution(readback_times);
        std::ofstream saved(prefix + ".json");
        saved << report.dump(2);
        std::cout << report.dump() << std::endl;
        ggml_backend_buffer_free(compute_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(context); ggml_free(weight_context); ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
