// Correctness probe: weight bytes come from the worker's indexed GGUF, then a
// matrix multiplication runs on the selected remote backend.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 3 && argc != 4 && argc != 5 && argc != 6) throw std::runtime_error("Usage: qw-rpc-probe loopback-endpoint device-index");
        ggml_backend_load_all();
        auto registry = ggml_backend_reg_by_name("RPC");
        if (!registry) throw std::runtime_error("RPC backend unavailable");
        using AddServer = ggml_backend_reg_t (*)(const char *);
        auto add = reinterpret_cast<AddServer>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_rpc_add_server"));
        auto remote = add(argv[1]);
        if (!remote) throw std::runtime_error("Remote worker unavailable");
        const size_t device = std::stoul(argv[2]);
        if (device >= ggml_backend_reg_dev_count(remote)) throw std::runtime_error("Remote device index unavailable");
        auto backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(remote, device), nullptr);
        if (!backend) throw std::runtime_error("Cannot initialize selected remote backend");
        ggml_init_params parameters{8 * 1024 * 1024, nullptr, true};
        auto weight_context = ggml_init(parameters);
        const bool batched = argc == 6 && std::string(argv[5]) == "--batch";
        const int slabs = batched ? 3 : 1;
        auto weights = ggml_new_tensor_2d(weight_context, GGML_TYPE_F32, 256, 256 * slabs);
        ggml_set_name(weights, "probe.weight");
        auto weight_buffer = ggml_backend_alloc_ctx_tensors(weight_context, backend);
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(256 * 256);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(i % 17) / 17.0f;
        int identifier_requests = 0;
        double identifier_seconds = 0;
        if (argc >= 4 && std::string(argv[3]) == "--indexed") {
            using Fill = bool (*)(ggml_tensor *, uint64_t, size_t, size_t);
            auto fill = reinterpret_cast<Fill>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_rpc_tensor_set_hash"));
            identifier_requests = argc >= 5 ? std::stoi(argv[4]) : 1;
            if (identifier_requests < 1 || identifier_requests > 10000)
                throw std::runtime_error("Identifier request count must be 1..10000");
            const auto started = std::chrono::steady_clock::now();
            if (batched) {
                using Batch = bool (*)(ggml_tensor **, const uint64_t *, const size_t *, const size_t *, size_t);
                auto batch = reinterpret_cast<Batch>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_rpc_tensors_set_hash"));
                const size_t count = static_cast<size_t>(identifier_requests);
                std::vector<ggml_tensor *> destinations(count, weights);
                std::vector<uint64_t> hashes(count, UINT64_C(0x920f32fa08eee431));
                std::vector<size_t> offsets(count), sizes(count, values.size() * sizeof(float));
                for (size_t i = 0; i < count; ++i) offsets[i] = (i % slabs) * sizes[i];
                if (count < static_cast<size_t>(slabs) || !batch ||
                    !batch(destinations.data(), hashes.data(), offsets.data(), sizes.data(), count))
                    throw std::runtime_error("Identifier batch failed");
                // Invalid lengths and destination bounds must be rejected locally.
                if (batch(destinations.data(), hashes.data(), offsets.data(), sizes.data(), 0) ||
                    batch(destinations.data(), hashes.data(), offsets.data(), sizes.data(), 4097))
                    throw std::runtime_error("Invalid batch count was accepted");
                offsets[0] = ggml_nbytes(weights);
                if (batch(destinations.data(), hashes.data(), offsets.data(), sizes.data(), 1))
                    throw std::runtime_error("Out-of-bounds destination was accepted");
            } else {
                for (int repeat = 0; repeat < identifier_requests; ++repeat) {
                    if (!fill || !fill(weights, UINT64_C(0x920f32fa08eee431), 0, values.size() * sizeof(float)))
                        throw std::runtime_error("Identifier-only local weight load failed");
                }
            }
            identifier_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        } else {
        ggml_backend_tensor_set(weights, values.data(), 0, values.size() * sizeof(float));
        }
        auto compute_context = ggml_init(parameters);
        auto input = ggml_new_tensor_2d(compute_context, GGML_TYPE_F32, 256, 1);
        auto output = ggml_mul_mat(compute_context, weights, input);
        auto graph = ggml_new_graph_custom(compute_context, 64, false);
        ggml_build_forward_expand(graph, output);
        auto compute_buffer = ggml_backend_alloc_ctx_tensors(compute_context, backend);
        std::vector<float> ones(256, 1.0f), actual(256 * slabs);
        ggml_backend_tensor_set(input, ones.data(), 0, ones.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Remote matrix multiplication failed");
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        float maximum_error = 0;
        for (size_t row = 0; row < static_cast<size_t>(256 * slabs); ++row) {
            float expected = 0;
            for (size_t column = 0; column < 256; ++column) expected += values[(row % 256) * 256 + column];
            maximum_error = std::fmax(maximum_error, std::fabs(expected - actual[row]));
        }
        if (maximum_error > 0.002f) throw std::runtime_error("Remote GPU result differs from the reference");
        std::cout << "{\"state\":\"done\",\"weight_bytes\":" << values.size() * sizeof(float)
                  << ",\"maximum_error\":" << maximum_error
                  << ",\"batched\":" << (batched ? "true" : "false")
                  << ",\"verified_slabs\":" << slabs
                  << ",\"identifier_requests\":" << identifier_requests
                  << ",\"identifier_seconds\":" << identifier_seconds << "}" << std::endl;
        ggml_backend_buffer_free(compute_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(compute_context);
        ggml_free(weight_context);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
