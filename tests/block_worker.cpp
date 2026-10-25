// Isolated complete-layer replay gate. Reuses Runtime's real operators/state.
// No ggml RPC backend is registered: only complete block inputs/outputs cross.
#include "model.h"
#include "runtime.h"
#include "tokenizer.h"
#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using Json = nlohmann::json;
using Frame = questwend::Runtime::BlockFrame;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

static void memory_guard() {
#ifdef _WIN32
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (!GlobalMemoryStatusEx(&memory) || memory.ullAvailPhys < (2ull << 30))
        throw std::runtime_error("block gate requires at least 2 GiB available system RAM");
#endif
}

static Json frame_json(const Frame & frame) {
    return {{"first", frame.first}, {"last", frame.last}, {"token", frame.token},
        {"position", frame.position}, {"rope_position", frame.rope_position},
        {"hidden", frame.hidden}, {"ple", frame.ple}, {"output", frame.output},
        {"elapsed_ms", frame.elapsed_ms}};
}

static Frame json_frame(const Json & value) {
    Frame frame;
    frame.first = value.at("first"); frame.last = value.at("last");
    frame.token = value.at("token"); frame.position = value.at("position");
    frame.rope_position = value.at("rope_position");
    frame.hidden = value.at("hidden").get<std::vector<float>>();
    frame.ple = value.at("ple").get<std::vector<float>>();
    frame.output = value.value("output", std::vector<float>{});
    frame.elapsed_ms = value.value("elapsed_ms", 0.0);
    return frame;
}

static Frame binary_frame(const std::string & body, size_t hidden_limit, size_t ple_limit) {
    // Both experiment nodes are little-endian Windows x86. Protocol is explicit
    // and versioned; a future portable implementation must encode byte order.
    std::array<uint32_t, 9> header{};
    if (body.size() < sizeof(header)) throw std::runtime_error("truncated block header");
    std::memcpy(header.data(), body.data(), sizeof(header));
    if (header[0] != 0x51424c4b || header[1] != 1)
        throw std::runtime_error("block protocol/version mismatch");
    const size_t hidden_count = header[7], ple_count = header[8];
    if (hidden_count != hidden_limit || ple_count != ple_limit ||
        body.size() != sizeof(header) + (hidden_count + ple_count) * sizeof(float))
        throw std::runtime_error("block frame length mismatch");
    Frame frame;
    frame.first = header[2]; frame.last = header[3]; frame.token = header[4];
    frame.position = header[5]; frame.rope_position = header[6];
    frame.hidden.resize(hidden_count); frame.ple.resize(ple_count);
    std::memcpy(frame.hidden.data(), body.data() + sizeof(header), hidden_count * sizeof(float));
    if (ple_count) std::memcpy(frame.ple.data(),
        body.data() + sizeof(header) + hidden_count * sizeof(float), ple_count * sizeof(float));
    return frame;
}

static Json compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("block output shape mismatch");
    double error = 0, magnitude = 0, max_abs = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error("non-finite block output");
        const double difference = (double) actual[i] - expected[i];
        error += difference * difference;
        magnitude += (double) expected[i] * expected[i];
        max_abs = std::max(max_abs, std::abs(difference));
    }
    return {{"relative_l2", std::sqrt(error / std::max(1e-30, magnitude))},
        {"max_abs", max_abs}, {"count", actual.size()}};
}

struct Options {
    std::string model, mode, input, output, state, remote_url, reference;
    int first = 0, last = 4, context = 512, max_tokens = 64, port = 8194, passes = 2;
    int prefill_tokens = 0;
    int output_tokens = 128;
    bool resident_ram = false, cpu_experts = false;
    size_t vram_mib = 4608;
};

static Options options(int argc, char ** argv) {
    Options result;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("every option requires a value");
        std::string key = argv[i], value = argv[i + 1];
        if (key == "--model") result.model = value;
        else if (key == "--mode") result.mode = value;
        else if (key == "--input") result.input = value;
        else if (key == "--output") result.output = value;
        else if (key == "--state") result.state = value;
        else if (key == "--remote-url") result.remote_url = value;
        else if (key == "--reference") result.reference = value;
        else if (key == "--output-tokens") result.output_tokens = std::stoi(value);
        else if (key == "--resident-ram") result.resident_ram = std::stoi(value) != 0;
        else if (key == "--cpu-experts") result.cpu_experts = std::stoi(value) != 0;
        else if (key == "--first") result.first = std::stoi(value);
        else if (key == "--last") result.last = std::stoi(value);
        else if (key == "--context") result.context = std::stoi(value);
        else if (key == "--max-tokens") result.max_tokens = std::stoi(value);
        else if (key == "--prefill-tokens") result.prefill_tokens = std::stoi(value);
        else if (key == "--port") result.port = std::stoi(value);
        else if (key == "--passes") result.passes = std::stoi(value);
        else if (key == "--vram-mib") result.vram_mib = std::stoull(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (result.model.empty() || result.first < 0 || result.last <= result.first ||
        result.context <= 0 || result.passes < 1 || result.max_tokens < 1 || result.prefill_tokens < 0)
        throw std::runtime_error("invalid model/range/context/pass options");
    return result;
}

static Json read_json(const std::string & path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path);
    Json value; stream >> value; return value;
}

static void write_json(const std::string & path, const Json & value) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write " + path);
    stream << value.dump(2);
    stream.flush();
    if (!stream) throw std::runtime_error("failed writing " + path);
}

static void load_worker_state(questwend::Runtime & runtime, const std::string & path) {
    if (path.empty()) { runtime.reset(); return; }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open owned-layer state: " + path);
    runtime.load_block_state([&](void * data, size_t bytes) {
        stream.read(static_cast<char *>(data), (std::streamsize) bytes);
        if (!stream) throw std::runtime_error("truncated owned-layer state");
    });
    if (stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("owned-layer state contains trailing bytes");
}

static std::string binary_request(const Frame & frame) {
    std::array<uint32_t, 9> header{{0x51424c4b, 1, (uint32_t) frame.first,
        (uint32_t) frame.last, (uint32_t) frame.token, (uint32_t) frame.position,
        (uint32_t) frame.rope_position, (uint32_t) frame.hidden.size(), (uint32_t) frame.ple.size()}};
    const size_t hidden_bytes = frame.hidden.size() * sizeof(float);
    const size_t ple_bytes = frame.ple.size() * sizeof(float);
    std::string body(sizeof(header) + hidden_bytes + ple_bytes, '\0');
    std::memcpy(body.data(), header.data(), sizeof(header));
    std::memcpy(body.data() + sizeof(header), frame.hidden.data(), hidden_bytes);
    if (ple_bytes) std::memcpy(body.data() + sizeof(header) + hidden_bytes, frame.ple.data(), ple_bytes);
    return body;
}

static int generation_comparison(const Options & option, questwend::Model & model,
        questwend::Runtime & runtime, double load_ms) {
    if (option.output_tokens < 1 || option.prefill_tokens < 1 ||
        option.prefill_tokens + option.output_tokens > option.context)
        throw std::runtime_error("control/relay needs positive prefill/output lengths within context");
    std::ifstream input(option.input, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read common coding input");
    std::string prompt((std::istreambuf_iterator<char>(input)), {});
    questwend::Tokenizer tokenizer(model.vocab());
    auto tokens = tokenizer.encode(prompt, tokenizer.wants_bos());
    if (tokens.size() < (size_t) option.prefill_tokens)
        throw std::runtime_error("common source prompt is shorter than requested prefix");
    tokens.resize((size_t) option.prefill_tokens);
    const auto prefill_started = Clock::now();
    std::vector<float> logits = runtime.decode(tokens);
    const double prefill_ms = elapsed_ms(prefill_started);
    std::vector<int32_t> reference;
    if (!option.reference.empty()) reference = read_json(option.reference).at("generated_tokens").get<std::vector<int32_t>>();
    if (!reference.empty() && reference.size() < (size_t) option.output_tokens)
        throw std::runtime_error("reference continuation has too few tokens");
    Json remote_rows = Json::array();
    std::unique_ptr<httplib::Client> client;
    if (option.mode == "relay") {
        if (option.remote_url.empty()) throw std::runtime_error("relay requires --remote-url");
        client = std::make_unique<httplib::Client>(option.remote_url);
        client->set_keep_alive(true); client->set_tcp_nodelay(true);
        client->set_connection_timeout(5); client->set_read_timeout(120);
        auto reset = client->Post("/reset", "", "application/octet-stream");
        auto health = client->Get("/health");
        if (!reset || reset->status != 200 || !health || health->status != 200)
            throw std::runtime_error("remote worker reset/health failed");
        const auto metadata = Json::parse(health->body);
        if (metadata.at("position") != option.prefill_tokens || metadata.at("first") != option.first ||
            metadata.at("last") != option.last || metadata.at("hidden_floats") != model.hparams().n_embd_hc())
            throw std::runtime_error("worker block/position/residual shape does not match primary prefix");
        runtime.execute_remote_block(option.first, option.last, [&](const Frame & frame) {
            const auto started = Clock::now();
            const auto body = binary_request(frame);
            auto reply = client->Post("/step", {{"Expect", ""}}, body, "application/octet-stream");
            if (!reply || reply->status != 200)
                throw std::runtime_error("remote block failed; session requires reset and replay");
            const double wall_ms = elapsed_ms(started);
            if (reply->body.size() != frame.hidden.size() * sizeof(float))
                throw std::runtime_error("remote block response length mismatch");
            const double worker_ms = std::stod(reply->get_header_value("X-Block-Wall-Ms"));
            std::vector<float> output(frame.hidden.size());
            std::memcpy(output.data(), reply->body.data(), reply->body.size());
            remote_rows.push_back({{"position", frame.position}, {"wall_ms", wall_ms},
                {"worker_ms", worker_ms}, {"transport_and_http_ms", wall_ms - worker_ms},
                {"request_bytes", body.size()}, {"response_bytes", reply->body.size()}});
            return output;
        });
    }
    Json result = {{"state", "running"}, {"mode", option.mode}, {"load_ms", load_ms},
        {"prefill_tokens", option.prefill_tokens}, {"prefill_ms", prefill_ms},
        {"teacher_forced_reference", !reference.empty()}, {"rows", Json::array()},
        {"generated_tokens", Json::array()}, {"remote_rows", remote_rows}};
    bool matches = true;
    double total_decode_ms = 0;
    for (int step = 0; step < option.output_tokens; ++step) {
        memory_guard();
        const int32_t predicted = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        const int32_t token = reference.empty() ? predicted : reference[(size_t) step];
        const bool matched = token == predicted;
        matches = matches && matched;
        const auto started = Clock::now();
        logits = runtime.decode({token});
        const double wall_ms = elapsed_ms(started);
        total_decode_ms += wall_ms;
        result["generated_tokens"].push_back(token);
        result["rows"].push_back({{"step", step}, {"position", runtime.n_past() - 1},
            {"token", token}, {"predicted", predicted}, {"greedy_match", matched}, {"decode_ms", wall_ms}});
        if ((step + 1) % 16 == 0) std::cerr << option.mode << " decoded steps: " << step + 1 << std::endl;
    }
    result["remote_rows"] = remote_rows;
    result["decode_ms"] = total_decode_ms;
    result["decode_tokens_per_second"] = 1000.0 * option.output_tokens / total_decode_ms;
    result["greedy_matches_reference"] = matches;
    result["state"] = matches ? "done" : "greedy_equivalence_failed";
    write_json(option.output, result);
    return matches ? 0 : 2;
}

int main(int argc, char ** argv) {
    try {
        const auto option = options(argc, argv);
        memory_guard();
        const auto load_started = Clock::now();
        auto model = questwend::Model::load(option.model);
        questwend::RuntimeConfig config;
        config.n_ctx = option.context; config.n_threads = 4;
        config.vram_budget_mb = option.vram_mib;
        config.expert_tier = option.resident_ram ? questwend::ExpertTier::Ram : questwend::ExpertTier::Ssd;
        config.block_cpu_experts = option.cpu_experts;
        if (option.mode == "capture" && (option.resident_ram || option.cpu_experts))
            throw std::runtime_error("capture reference must use original GPU/SSD model path");
        config.ngram_mode = "disk"; config.ngram_cache_mb = 256;
        config.cache_profile_save = false;
        if (option.mode == "serve" || option.mode == "replay") {
            config.block_first = option.first; config.block_last = option.last;
        }
        questwend::Runtime runtime(*model, config);
        const double load_ms = elapsed_ms(load_started);
        if (option.mode == "control" || option.mode == "relay")
            return generation_comparison(option, *model, runtime, load_ms);
        if (option.mode == "capture") {
            Json trace = {{"schema", "questwend-complete-block-v1"},
                {"model", option.model}, {"first", option.first}, {"last", option.last},
                {"context", option.context}, {"load_ms", load_ms}, {"frames", Json::array()}};
            std::ifstream stream(option.input, std::ios::binary);
            if (!stream) throw std::runtime_error("cannot read source prompt");
            std::string prompt((std::istreambuf_iterator<char>(stream)), {});
            questwend::Tokenizer tokenizer(model->vocab());
            auto tokens = tokenizer.encode(prompt, tokenizer.wants_bos());
            tokens.resize(std::min(tokens.size(), (size_t) option.prefill_tokens + option.max_tokens));
            if (tokens.empty()) throw std::runtime_error("capture prompt tokenized empty");
            if ((size_t) option.prefill_tokens >= tokens.size())
                throw std::runtime_error("prompt needs continuation tokens after common prefill");
            if (option.prefill_tokens) {
                if (option.state.empty()) throw std::runtime_error("medium-context capture needs --state output");
                const auto prefill_started = Clock::now();
                runtime.decode(std::vector<int32_t>(tokens.begin(), tokens.begin() + option.prefill_tokens));
                trace["prefill_ms"] = elapsed_ms(prefill_started);
                trace["prefill_tokens"] = option.prefill_tokens;
                memory_guard();
                std::ofstream state(option.state, std::ios::binary);
                if (!state) throw std::runtime_error("cannot create owned-layer state");
                runtime.save_block_state(option.first, option.last, [&](const void * data, size_t bytes) {
                    state.write(static_cast<const char *>(data), (std::streamsize) bytes);
                    if (!state) throw std::runtime_error("cannot write owned-layer state");
                });
                state.flush();
                if (!state) throw std::runtime_error("cannot flush owned-layer state");
                trace["state_file"] = option.state;
            }
            runtime.capture_block(option.first, option.last,
                [&](const Frame & frame) {
                    trace["frames"].push_back(frame_json(frame));
                    if (trace["frames"].size() % 16 == 0)
                        std::cerr << "captured block frames: " << trace["frames"].size() << std::endl;
                });
            for (size_t index = (size_t) option.prefill_tokens; index < tokens.size(); ++index) {
                memory_guard(); runtime.decode({tokens[index]});
            }
            trace["state"] = "done"; write_json(option.output, trace);
            return 0;
        }
        if (option.mode == "replay") {
            const auto trace = read_json(option.input);
            Json result = {{"state", "running"}, {"load_ms", load_ms},
                {"first", option.first}, {"last", option.last}, {"passes", Json::array()}};
            bool passed = true;
            for (int pass = 0; pass < option.passes; ++pass) {
                load_worker_state(runtime, option.state);
                Json rows = Json::array();
                for (const auto & value : trace.at("frames")) {
                    memory_guard();
                    const Frame frame = json_frame(value);
                    const auto started = Clock::now();
                    const auto & output = runtime.block_step(frame);
                    const double wall_ms = elapsed_ms(started);
                    auto row = compare(output, frame.output);
                    const double relative_l2 = row["relative_l2"];
                    // Initial cross-GPU gate. Preserve measured errors rather
                    // than claiming bitwise identity across CUDA architectures.
                    const bool correct = relative_l2 <= 0.005;
                    passed = passed && correct;
                    row["passed"] = correct; row["position"] = frame.position;
                    row["wall_ms"] = wall_ms; row["reference_block_ms"] = frame.elapsed_ms;
                    rows.push_back(row);
                }
                result["passes"].push_back({{"index", pass}, {"rows", rows}});
                write_json(option.output, result);
            }
            result["state"] = passed ? "done" : "numerical_gate_failed";
            result["passed"] = passed; write_json(option.output, result);
            return passed ? 0 : 2;
        }
        if (option.mode != "serve") throw std::runtime_error("mode must be capture, replay or serve");
        httplib::Server server;
        std::mutex session;
        server.set_tcp_nodelay(true);
        server.set_keep_alive_max_count(100000);
        server.set_keep_alive_timeout(300);
        server.set_payload_max_length(4 * 1024 * 1024);
        server.Get("/health", [&](const httplib::Request &, httplib::Response & response) {
            response.set_content(Json({{"ready", true}, {"first", option.first},
                {"last", option.last}, {"position", runtime.n_past()}, {"load_ms", load_ms},
                {"managed_resident_ram", option.resident_ram}, {"cpu_experts", option.cpu_experts},
                {"hidden_floats", model->hparams().n_embd_hc()},
                {"ple_floats", model->hparams().has_ple() ? model->hparams().n_embd : 0}}).dump(),
                "application/json");
        });
        server.Post("/reset", [&](const httplib::Request &, httplib::Response & response) {
            std::lock_guard<std::mutex> lock(session);
            try {
                load_worker_state(runtime, option.state);
                response.set_content("{}", "application/json");
            } catch (const std::exception & error) {
                response.status = 400;
                response.set_content(Json({{"error", error.what()}, {"reset_required", true}}).dump(),
                    "application/json");
            }
        });
        server.Post("/step", [&](const httplib::Request & request, httplib::Response & response) {
            std::lock_guard<std::mutex> lock(session);
            try {
                memory_guard();
                const auto frame = binary_frame(request.body, model->hparams().n_embd_hc(),
                    model->hparams().has_ple() ? model->hparams().n_embd : 0);
                const auto started = Clock::now();
                const auto & output = runtime.block_step(frame);
                response.set_header("X-Block-Wall-Ms", std::to_string(elapsed_ms(started)));
                response.set_content(reinterpret_cast<const char *>(output.data()),
                    output.size() * sizeof(float), "application/octet-stream");
            } catch (const std::exception & error) {
                // A partially advanced failure requires explicit reset/replay.
                response.status = 400;
                response.set_content(Json({{"error", error.what()}, {"reset_required", true}}).dump(),
                    "application/json");
            }
        });
        std::cerr << "block worker ready on loopback:" << option.port << std::endl;
        if (!server.listen("127.0.0.1", option.port)) throw std::runtime_error("cannot bind block worker port");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "block gate failed: " << error.what() << std::endl;
        return 1;
    }
}
