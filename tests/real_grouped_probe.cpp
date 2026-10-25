// Isolated full-model grouped expert correctness / timing gate. Configuration
// and artifact files are experiment-owned. It never manages gateway services.
#include "model.h"
#include "runtime.h"
#include "tokenizer.h"
#include "real_grouped_client.h"
#include "real_grouped_reference.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <windows.h>

using namespace grouped_gate;

static void memory_guard() {
    MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
    if (!GlobalMemoryStatusEx(&memory) || memory.ullAvailPhys < (2ull << 30))
        throw std::runtime_error("Real grouped gate requires at least2 GiB available RAM");
}

static Json read_json(const std::string & path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot read JSON: " + path);
    Json value; stream >> value; return value;
}

static void write_json(const std::string & path, const Json & value) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot write JSON: " + path);
    stream << value.dump(2); stream.flush();
    if (!stream) throw std::runtime_error("Result write failed");
}

static void write_floats(const std::string & path, const std::vector<float> & values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    if (!stream) throw std::runtime_error("Cannot preserve numerical vector");
}

static int argmax(const std::vector<float> & logits) {
    if (logits.empty()) throw std::runtime_error("Empty model logits");
    for (float value : logits) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite model logits");
    return (int) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

static void import_seed(questwend::Runtime & runtime, const std::string & path) {
    std::ifstream state(path, std::ios::binary);
    if (!state) throw std::runtime_error("Cannot read complete probe snapshot");
    runtime.load_probe_state([&](void * data, size_t bytes) {
        state.read(static_cast<char *>(data), bytes);
        if (!state) throw std::runtime_error("Truncated complete probe snapshot");
    });
    if (state.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unconsumed probe state bytes");
}

int main(int argc, char ** argv) {
    Json result{{"state", "starting"}};
    std::string output;
    try {
        if (argc != 2) throw std::runtime_error("Usage: qw-real-grouped-probe config.json");
        const Json options = read_json(argv[1]);
        output = options.at("output");
        const std::string mode = options.at("mode");
        const bool validation = mode == "validate";
        if (mode != "reference" && mode != "validate" && mode != "measure")
            throw std::runtime_error("Mode must be reference, validate or measure");
        const int steps = options.value("tokens", 32), remote_count = options.value("remote_count", 4);
        const int warmup_steps = options.value("warmup_steps", 0);
        if (steps < 1 || steps > 64 || remote_count < 1 || remote_count > 4)
            throw std::runtime_error("Tiny gate is bounded at64 generated tokens and4 remote experts/layer");
        if (warmup_steps < 0 || warmup_steps > steps || (warmup_steps && !options.contains("state_file")))
            throw std::runtime_error("Seeded cache warmup needs a saved probe state and0..steps token count");
        const Json manifest = read_json(options.at("corpus"));
        const std::vector<int> layers = options.at("layers").get<std::vector<int>>();
        if (layers.empty() || layers.size() > 4) throw std::runtime_error("Select one tofour validated expert layers");
        for (int layer : layers) {
            const auto & supported = manifest.at("signatures").at(0).at("available_layers");
            if (std::find(supported.begin(), supported.end(), Json(layer)) == supported.end())
                throw std::runtime_error("Requested layer is absent from resident manifest");
        }
        _putenv_s("QWEN_CPU_MISS", "0"); // Never run automatic CPU calibration.
        memory_guard();
        const auto load_started = Clock::now();
        auto model = questwend::Model::load(options.at("model"));
        questwend::RuntimeConfig config;
        config.n_ctx = options.value("context", 512);
        config.n_threads = options.value("threads", 4);
        config.vram_budget_mb = options.value("vram_mib", size_t(21504));
        config.expert_tier = questwend::ExpertTier::Ssd;
        config.ngram_mode = options.value("ngram_mode", std::string("disk"));
        config.ngram_cache_mb = options.value("ngram_cache_mib", size_t(4096));
        config.cache_profile_save = false;
        questwend::Runtime runtime(*model, config);
        questwend::Tokenizer tokenizer(model->vocab());
        result = {{"state", "running"}, {"mode", mode}, {"load_ms", elapsed_ms(load_started)},
            {"model_fingerprint", manifest.at("model_fingerprint")}, {"model", options.at("model")},
            {"input_identity", options.at("input_identity")},
            {"warmup_steps", warmup_steps},
            {"cache_state_note", "Inference snapshot excludes expert cache residency. Each fresh process starts its own cache; optional seed replay warms before measured decode."},
            {"layers", layers}, {"context", config.n_ctx}, {"vram_mib", config.vram_budget_mb},
            {"ngram_mode", config.ngram_mode}, {"ngram_cache_mib", config.ngram_cache_mb},
            {"remote_count", mode == "reference" ? 0 : remote_count},
            {"misses_only", options.value("misses_only", false)},
            {"ssd_misses_only", options.value("ssd_misses_only", false)},
            {"tolerance_unchanged", 0.03}, {"expert_checks", Json::array()}, {"turns", Json::array()},
            {"timings_include_shadow_work", validation},
            {"scope", "Actual full-model routing; experimental fixedsplit or GPU-cache-miss-only split. All attention/KV/GDN/HC state stays primary. Not a capacity-placement proof."}};
        int pending = -1;
        const auto prefill_started = Clock::now();
        if (options.contains("state_file")) {
            import_seed(runtime, options.at("state_file"));
            pending = options.at("pending");
            result["state_file"] = options.at("state_file");
        } else {
            std::ifstream stream(options.at("prompt_file").get<std::string>(), std::ios::binary);
            if (!stream) throw std::runtime_error("Cannot read raw prompt file");
            const std::string prompt((std::istreambuf_iterator<char>(stream)), {});
            auto tokens = tokenizer.encode(prompt, tokenizer.wants_bos());
            if (tokens.empty() || tokens.size() + steps >= size_t(config.n_ctx))
                throw std::runtime_error("Prompt empty or exceeds configured context");
            pending = tokens.back(); tokens.pop_back();
            if (!tokens.empty()) runtime.decode(tokens);
            result["prompt_tokens"] = tokens.size() + 1;
        }
        result["prefill_or_import_ms"] = elapsed_ms(prefill_started);
        result["start_position"] = runtime.n_past(); result["pending_input"] = pending;
        if (pending < 0 || runtime.n_past() + steps >= config.n_ctx)
            throw std::runtime_error("Pending token/context headroom invalid");
        Json reference;
        if (mode != "reference") {
            reference = read_json(options.at("reference"));
            for (const char * key : {"model_fingerprint", "layers", "context", "vram_mib", "ngram_mode",
                 "ngram_cache_mib", "start_position", "pending_input", "input_identity", "warmup_steps"})
                if (reference.at(key) != result.at(key)) throw std::runtime_error(std::string("Reference identity differs: ") + key);
            if (reference.at("mode") != "reference" || reference.at("state") != "done" || reference.at("turns").size() < size_t(steps))
                throw std::runtime_error("Incomplete all-local reference");
            if (mode == "measure") {
                const auto check = read_json(options.at("validated_gate"));
                if (check.at("state") != "done" || check.at("mode") != "validate" ||
                    check.at("model_fingerprint") != result.at("model_fingerprint") ||
                    check.at("layers") != result.at("layers") || check.at("context") != result.at("context") ||
                    check.at("remote_count") != remote_count || check.at("misses_only") != result.at("misses_only") ||
                    check.at("ssd_misses_only") != result.at("ssd_misses_only"))
                    throw std::runtime_error("Measurement requires a matching successful numerical gate");
            }
        }
        std::unique_ptr<RemoteClient> remote;
        std::unique_ptr<NativeReference> numerical;
        questwend::Runtime::GroupedExpertExecutor execute;
        if (mode != "reference") {
            remote.reset(new RemoteClient(options.value("port", 50322), options.at("experiment_key"), manifest.at("model_fingerprint")));
            result["worker_identity"] = remote->identity();
            if (validation) numerical.reset(new NativeReference(manifest));
            execute = [&](const Request & request) {
                memory_guard();
                auto output_values = remote->execute(request);
                result["remote_calls"] = remote->calls;
                if (validation) {
                    const auto expected = numerical->execute(request);
                    auto comparison = compare(expected, output_values);
                    comparison["layer"] = request.layer; comparison["position"] = request.position;
                    comparison["experts"] = request.experts; comparison["weights"] = request.weights;
                    result["expert_checks"].push_back(comparison);
                    if (comparison.at("relative_l2").get<double>() > .03) {
                        const std::string prefix = output + ".failed-l" + std::to_string(request.layer) + "-p" + std::to_string(request.position);
                        write_floats(prefix + ".input.f32", request.input);
                        write_floats(prefix + ".remote.f32", output_values);
                        write_floats(prefix + ".native-gpu.f32", expected);
                        result["failed_expert_vectors"] = prefix;
                        throw std::runtime_error("Real routed expert contribution exceeds unchanged3% numerical gate");
                    }
                }
                return output_values;
            };
        }
        runtime.configure_grouped_experts(layers, mode == "reference" ? 0 : remote_count, execute,
            options.value("misses_only", false), options.value("ssd_misses_only", false));
        if (warmup_steps) {
            int warm_pending = pending;
            const auto started = Clock::now();
            for (int step = 0; step < warmup_steps; ++step) {
                memory_guard();
                const int next = argmax(runtime.decode({warm_pending}));
                // Same known token path for each arm, rather than silently
                // warming a divergent conversation. Gate failures stay failures.
                if (mode != "reference" && next != reference.at("turns").at(step).at("next").get<int>())
                    throw std::runtime_error("Warmup greedy token differs from all-local control");
                warm_pending = next;
            }
            result["cache_warmup_ms"] = elapsed_ms(started);
            if (remote) { result["warmup_remote_calls"] = remote->calls; remote->calls = Json::array(); }
            result["warmup_expert_checks"] = result["expert_checks"];
            result["expert_checks"] = Json::array();
            import_seed(runtime, options.at("state_file"));
            runtime.configure_grouped_experts(layers, mode == "reference" ? 0 : remote_count, execute,
                options.value("misses_only", false), options.value("ssd_misses_only", false));
            memory_guard();
        }
        const auto generation_started = Clock::now();
        double decode_total = 0;
        for (int step = 0; step < steps; ++step) {
            memory_guard();
            const auto started = Clock::now();
            const auto & logits = runtime.decode({pending});
            const double wall = elapsed_ms(started);
            decode_total += wall;
            const int next = argmax(logits);
            Json row{{"step", step}, {"input", pending}, {"next", next}, {"decode_ms", wall}};
            if (mode == "reference") {
                const std::string path = output + ".logits-" + std::to_string(step) + ".f32";
                write_floats(path, logits); row["logits_file"] = path;
            } else {
                const auto & expected_row = reference.at("turns").at(step);
                std::ifstream stream(expected_row.at("logits_file").get<std::string>(), std::ios::binary);
                std::vector<float> expected(logits.size());
                stream.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
                if (!stream || stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Reference logits shape mismatch");
                row["logits_difference"] = compare(expected, logits);
                row["greedy_matches"] = next == expected_row.at("next").get<int>();
                if (!row["greedy_matches"].get<bool>()) {
                    result["turns"].push_back(row);
                    write_floats(output + ".failed-logits.f32", logits);
                    throw std::runtime_error("Full-model greedy token differs from all-local control");
                }
            }
            result["turns"].push_back(row); pending = next;
            write_json(output, result);
            std::cout << Json{{"step", step}, {"decode_ms", wall}, {"next", next}}.dump() << std::endl;
        }
        result["generation_wall_ms"] = elapsed_ms(generation_started);
        result["decode_total_ms"] = decode_total;
        result["generation_tps"] = validation ? Json(nullptr) : Json(steps * 1000.0 / decode_total);
        result["pending_after"] = pending;
        if (remote) result["remote_calls"] = remote->calls;
        if (mode != "reference" && (!remote || remote->calls.empty()))
            throw std::runtime_error("No actual remote calls occurred; gate cannot validate an unused executor");
        result["state"] = "done"; write_json(output, result);
        return 0;
    } catch (const std::exception & error) {
        result["state"] = "failed"; result["error"] = error.what();
        if (!output.empty()) write_json(output, result);
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
