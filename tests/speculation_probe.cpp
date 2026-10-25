// Isolated JSON-lines probe. All model execution uses the existing runtime;
// no HTTP/gateway configuration is changed. stderr retains native diagnostics.
#include "args.h"
#include "chat.h"
#include "model.h"
#include "runtime.h"
#include "tokenizer.h"
#include "json.hpp"
#include "httplib.h"
#include <array>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;
using namespace questwend;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static int32_t argmax(const std::vector<float> & logits) {
    if (logits.empty()) throw std::runtime_error("No logits are available");
    return (int32_t) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

// Shared with free generation: exactly one ordinary single-token decode,
// including greedy argmax in the native per-step timing.
static void decode_one_with_timing(Runtime & runtime, int32_t input,
                                   int32_t & pending, std::vector<double> & times) {
    const auto step = Clock::now();
    pending = argmax(runtime.decode({input}));
    times.push_back(elapsed_ms(step));
}

static json verification_json(const Runtime::VerificationProbe & value) {
    return {{"next_tokens", value.next_tokens}, {"input_tokens", value.input_tokens},
        {"kept_tokens", value.kept_tokens}, {"accepted_drafts", value.accepted_drafts},
        {"next_token", value.next_token}, {"batched", value.batched},
        {"replayed_for_state", value.replayed_for_state},
        {"prepare_ms", value.prepare_ms}, {"verify_ms", value.verify_ms},
        {"settle_ms", value.settle_ms}};
}


static std::string relay_binary_request(const Runtime::BlockFrame & frame) {
    const std::array<uint32_t, 9> header{{0x51424c4b, 1, (uint32_t) frame.first,
        (uint32_t) frame.last, (uint32_t) frame.token, (uint32_t) frame.position,
        (uint32_t) frame.rope_position, (uint32_t) frame.hidden.size(), (uint32_t) frame.ple.size()}};
    const size_t hidden_bytes = frame.hidden.size() * sizeof(float);
    const size_t ple_bytes = frame.ple.size() * sizeof(float);
    std::string body(sizeof(header) + hidden_bytes + ple_bytes, char(0));
    std::memcpy(body.data(), header.data(), sizeof(header));
    std::memcpy(body.data() + sizeof(header), frame.hidden.data(), hidden_bytes);
    if (ple_bytes) std::memcpy(body.data() + sizeof(header) + hidden_bytes, frame.ple.data(), ple_bytes);
    return body;
}

int main(int argc, char ** argv) {
    try {
        CommonOptions options;
        options.reasoning = false;
        int threads = 8;
        std::string json_input_path, json_output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--json-input" && i + 1 < argc) {
                json_input_path = argv[++i];
            } else if (argument == "--json-output" && i + 1 < argc) {
                json_output_path = argv[++i];
            } else if (argument == "--threads" && i + 1 < argc) {
                threads = std::stoi(argv[++i]);
            } else if (parse_common_arg(argc, argv, i, options) != ArgResult::Consumed) {
                throw std::runtime_error("Unknown/malformed argument: " + argument);
            }
        }
        // Dedicated profiler transport only. Ordinary stdio remains unchanged.
        if (json_input_path.empty() != json_output_path.empty())
            throw std::runtime_error("--json-input and --json-output must be supplied together");
        std::ifstream json_input_file;
        std::ofstream json_output_file;
        std::istream * json_input = &std::cin;
        std::ostream * json_output = &std::cout;
        if (!json_input_path.empty()) {
            if (std::filesystem::exists(json_output_path) &&
                std::filesystem::equivalent(json_input_path, json_output_path))
                throw std::runtime_error("JSON request and response files must differ");
            json_input_file.open(json_input_path);
            if (!json_input_file) throw std::runtime_error("Cannot open JSON request file");
            json_output_file.open(json_output_path, std::ios::out | std::ios::trunc);
            if (!json_output_file) throw std::runtime_error("Cannot open JSON response file");
            json_output_file.exceptions(std::ios::badbit | std::ios::failbit);
            json_input = &json_input_file;
            json_output = &json_output_file;
        }
        if (!validate_common_options(options)) return 1;
        if (options.use_mtp) throw std::runtime_error("Probe does not enable MTP");
        auto model = Model::load(options.model_path); // metadata only until prefill
        Tokenizer tokenizer(model->vocab());
        RuntimeConfig config;
        config.n_threads = threads;
        config.n_ctx = options.n_ctx > 0 ? options.n_ctx : (int) model->hparams().n_ctx_train;
        if (!apply_common_options(options, config)) return 1;
        std::unique_ptr<Runtime> runtime;
        std::vector<uint8_t> snapshot;
        int32_t pending = -1, saved_pending = -1;
        int checkpoint_position = -1;
        int32_t checkpoint_pending = -1;
        std::unique_ptr<httplib::Client> relay_client;
        json relay_rows = json::array();
        json capture_frames = json::array();
        std::string capture_file;
        int capture_first = -1, capture_last = -1, capture_prefill_tokens = 0;
        *json_output << json({{"state", "metadata_ready"}, {"vocab_size", model->vocab().tokens.size()},
            {"model", options.model_path}, {"context", config.n_ctx}}).dump() << std::endl;

        std::string line;
        while (std::getline(*json_input, line)) {
            if (line.empty()) continue;
            json response;
            try {
                const auto request = json::parse(line);
                const auto operation = request.at("op").get<std::string>();
                const auto started = Clock::now();
                if (operation == "quit") {
                    *json_output << json({{"ok", true}, {"op", operation}}).dump() << std::endl;
                    break;
                }
                if (operation == "render") {
                    std::vector<ChatMessage> messages;
                    for (const auto & input : request.at("messages")) {
                        ChatMessage message;
                        message.role = input.at("role").get<std::string>();
                        message.content = input.at("content").get<std::string>();
                        messages.push_back(std::move(message));
                    }
                    ChatPromptOptions chat;
                    chat.reasoning = request.value("reasoning", false);
                    chat.preserve_thinking = true;
                    const auto prompt = build_qwen_prompt(tokenizer, messages, chat);
                    response = {{"tokens", prompt.ids}, {"count", prompt.ids.size()},
                                {"generation_prompt_begin", prompt.gen_prompt_begin}};
                } else if (operation == "tokenize") {
                    const auto tokens = tokenizer.encode(request.at("text").get<std::string>(), false);
                    response = {{"tokens", tokens}, {"count", tokens.size()}};
                } else if (operation == "detokenize") {
                    response = {{"text", tokenizer.decode(request.at("tokens").get<std::vector<int32_t>>())}};
                } else {
                    if (!runtime) {
                        const auto load_started = Clock::now();
                        runtime = std::make_unique<Runtime>(*model, config);
                        runtime->set_progress_cb([](int done, int total) {
                            std::cerr << "speculation-prefill " << done << "/" << total << std::endl;
                        });
                        std::cerr << "speculation-probe: runtime loaded in " << elapsed_ms(load_started)
                                  << " ms" << std::endl;
                    }
                    if (operation == "prefill") {
                        const auto tokens = request.at("tokens").get<std::vector<int32_t>>();
                        if (tokens.empty()) throw std::runtime_error("Empty prefill");
                        if (request.value("reset", true)) runtime->reset();
                        pending = argmax(runtime->decode(tokens));
                        response = {{"tokens", tokens.size()}, {"pending", pending}};
                    } else if (operation == "decode") {
                        const auto tokens = request.at("tokens").get<std::vector<int32_t>>();
                        if (tokens.empty()) throw std::runtime_error("Empty decode");
                        const auto & logits = runtime->decode(tokens);
                        pending = argmax(logits);
                        std::vector<int32_t> sorted(logits.size());
                        for (size_t i = 0; i < sorted.size(); ++i) sorted[i] = (int32_t) i;
                        const int top = std::min((int) sorted.size(), request.value("top", 2));
                        std::partial_sort(sorted.begin(), sorted.begin() + top, sorted.end(),
                            [&](int32_t left, int32_t right) { return logits[left] > logits[right]; });
                        json ranking = json::array();
                        for (int i = 0; i < top; ++i)
                            ranking.push_back({{"id", sorted[i]}, {"logit", logits[sorted[i]]}});
                        json selected = json::array();
                        if (request.contains("score_ids")) {
                            for (int32_t token : request.at("score_ids").get<std::vector<int32_t>>()) {
                                if (token < 0 || token >= (int32_t) logits.size())
                                    throw std::runtime_error("Selected logit outside vocabulary");
                                selected.push_back({{"id", token}, {"logit", logits[token]}});
                            }
                        }
                        response = {{"pending", pending}, {"top", ranking}, {"selected_logits", selected}};
                    } else if (operation == "generate") {
                        const int count = request.value("count", 32);
                        if (count < 1 || count > 1024 || pending < 0)
                            throw std::runtime_error("Invalid generation count/pending token");
                        std::vector<int32_t> emitted;
                        std::vector<double> times;
                        for (int i = 0; i < count; ++i) {
                            emitted.push_back(pending);
                            decode_one_with_timing(*runtime, pending, pending, times);
                        }
                        response = {{"tokens", emitted}, {"text", tokenizer.decode(emitted)},
                            {"pending", pending}, {"decode_step_ms", times},
                            {"decode_forward_count", count}};
                    } else if (operation == "teacher_force") {
                        // A workload control, not free generation: the next input
                        // always comes from this fixed list, never from argmax.
                        const auto & requested = request.at("tokens");
                        if (!requested.is_array() || requested.empty() || requested.size() > 1024)
                            throw std::runtime_error("Teacher force needs 1..1024 exact token IDs");
                        const bool detokenize = request.value("detokenize", false);
                        const int start_position = runtime->n_past();
                        const size_t vocabulary = model->vocab().tokens.size();
                        if (pending < 0 || static_cast<size_t>(pending) >= vocabulary)
                            throw std::runtime_error("Teacher force requires a valid existing pending token");
                        if (start_position < 0 || start_position > config.n_ctx ||
                            requested.size() > static_cast<size_t>(config.n_ctx - start_position))
                            throw std::runtime_error("Teacher force would exceed the configured context");
                        std::vector<int32_t> inputs;
                        inputs.reserve(requested.size());
                        for (const auto & entry : requested) {
                            if (!entry.is_number_integer())
                                throw std::runtime_error("Teacher force token IDs must be integers");
                            const int64_t token = entry.get<int64_t>();
                            if (token < 0 || static_cast<uint64_t>(token) >= vocabulary)
                                throw std::runtime_error("Teacher force token outside vocabulary");
                            inputs.push_back(static_cast<int32_t>(token));
                        }
                        // All validation above completes before any state changes.
                        const int32_t initial_pending = pending;
                        std::vector<int32_t> predictions;
                        std::vector<int> positions;
                        std::vector<double> times;
                        predictions.reserve(inputs.size());
                        positions.reserve(inputs.size());
                        times.reserve(inputs.size());
                        for (int32_t input : inputs) {
                            positions.push_back(runtime->n_past());
                            decode_one_with_timing(*runtime, input, pending, times);
                            predictions.push_back(pending);
                        }
                        response = {{"benchmark_mode", "fixed_token_control"},
                            {"input_tokens", inputs}, {"input_positions", positions},
                            {"predicted_next_tokens", predictions},
                            {"initial_pending", initial_pending},
                            {"first_input_matches_initial_pending", inputs.front() == initial_pending},
                            {"pending", pending}, {"position_start", start_position},
                            {"position_end", runtime->n_past()}, {"context_limit", config.n_ctx},
                            {"decode_step_ms", times}, {"decode_forward_count", inputs.size()}};
                        if (detokenize) {
                            response["input_text"] = tokenizer.decode(inputs);
                            response["predicted_next_text"] = tokenizer.decode(predictions);
                        }
                    } else if (operation == "snapshot") {
                        snapshot.clear();
                        runtime->save_probe_state([&](const void * bytes, size_t size) {
                            if (snapshot.size() + size > (size_t) 2 * 1024 * 1024 * 1024)
                                throw std::runtime_error("Probe snapshot exceeds 2 GiB cap");
                            const auto * first = (const uint8_t *) bytes;
                            snapshot.insert(snapshot.end(), first, first + size);
                        });
                        saved_pending = pending;
                        response = {{"snapshot_bytes", snapshot.size()}, {"pending", pending}};
                    } else if (operation == "checkpoint") {
                        runtime->snapshot_ckpt();
                        checkpoint_position = runtime->n_past();
                        checkpoint_pending = pending;
                        response = {{"position", checkpoint_position}, {"pending", pending}};
                    } else if (operation == "rewind_checkpoint") {
                        if (checkpoint_position < 0) throw std::runtime_error("No prompt checkpoint");
                        const int position = runtime->rewind_to(checkpoint_position);
                        if (position != checkpoint_position)
                            throw std::runtime_error("Exact draft checkpoint no longer available");
                        pending = checkpoint_pending;
                        response = {{"position", position}, {"pending", pending}};
                    } else if (operation == "relay_enable") {
                        const int first = request.at("first").get<int>();
                        const int last = request.at("last").get<int>();
                        const std::string endpoint = request.at("endpoint").get<std::string>();
                        if (endpoint.rfind("http://127.0.0.1:", 0) != 0)
                            throw std::runtime_error("Relay endpoint must use the isolated loopback SSH route");
                        relay_client = std::make_unique<httplib::Client>(endpoint);
                        relay_client->set_keep_alive(true);
                        relay_client->set_tcp_nodelay(true);
                        relay_client->set_connection_timeout(5);
                        relay_client->set_read_timeout(120);
                        const auto reset = relay_client->Post("/reset", "", "application/octet-stream");
                        const auto health = relay_client->Get("/health");
                        if (!reset || reset->status != 200 || !health || health->status != 200)
                            throw std::runtime_error("Remote block reset/health failed");
                        const auto metadata = json::parse(health->body);
                        if (metadata.at("position") != runtime->n_past() ||
                            metadata.at("first") != first || metadata.at("last") != last ||
                            metadata.at("hidden_floats") != model->hparams().n_embd_hc())
                            throw std::runtime_error("Remote block state does not match the primary prefix");
                        relay_rows = json::array();
                        runtime->execute_remote_block(first, last, [&](const Runtime::BlockFrame & frame) {
                            const auto begin = Clock::now();
                            const auto body = relay_binary_request(frame);
                            const auto reply = relay_client->Post("/step", {{"Expect", ""}}, body, "application/octet-stream");
                            if (!reply || reply->status != 200)
                                throw std::runtime_error("Remote block failed; reset and restore primary state before reuse");
                            const double wall_ms = elapsed_ms(begin);
                            if (reply->body.size() != frame.hidden.size() * sizeof(float))
                                throw std::runtime_error("Remote block residual byte count mismatch");
                            const double worker_ms = std::stod(reply->get_header_value("X-Block-Wall-Ms"));
                            std::vector<float> output(frame.hidden.size());
                            std::memcpy(output.data(), reply->body.data(), reply->body.size());
                            relay_rows.push_back({{"position", frame.position}, {"wall_ms", wall_ms},
                                {"worker_ms", worker_ms}, {"transport_and_http_ms", wall_ms - worker_ms},
                                {"request_bytes", body.size()}, {"response_bytes", reply->body.size()}});
                            return output;
                        });
                        response = {{"remote", metadata}, {"endpoint", endpoint}};
                    } else if (operation == "relay_stats") {
                        response = {{"rows", relay_rows}, {"calls", relay_rows.size()}};
                    } else if (operation == "relay_disable") {
                        // The skipped primary layers are stale. Never merely
                        // remove a callback and continue from that invalid state.
                        runtime->reset(); // core reset also clears the executor.
                        relay_client.reset();
                        pending = -1;
                        response = {{"reset", true}, {"requires_probe_restore_or_prefill", true},
                                    {"rows", relay_rows}};
                    } else if (operation == "capture_block") {
                        capture_first = request.at("first").get<int>();
                        capture_last = request.at("last").get<int>();
                        capture_file = request.at("trace_file").get<std::string>();
                        capture_prefill_tokens = runtime->n_past();
                        capture_frames = json::array();
                        std::ofstream state(request.at("state_file").get<std::string>(), std::ios::binary);
                        if (!state) throw std::runtime_error("Cannot create owned-layer snapshot");
                        runtime->save_block_state(capture_first, capture_last,
                            [&](const void * bytes, size_t size) {
                                state.write((const char *) bytes, size);
                                if (!state) throw std::runtime_error("Owned-layer snapshot write failed");
                            });
                        runtime->capture_block(capture_first, capture_last,
                            [&](const Runtime::BlockFrame & frame) {
                                if (capture_frames.size() >= 128)
                                    throw std::runtime_error("Block capture exceeds 128-frame cap");
                                capture_frames.push_back({{"first", frame.first}, {"last", frame.last},
                                    {"token", frame.token}, {"position", frame.position},
                                    {"rope_position", frame.rope_position}, {"hidden", frame.hidden},
                                    {"ple", frame.ple}, {"output", frame.output}, {"elapsed_ms", frame.elapsed_ms}});
                            });
                        response = {{"capturing", true}, {"first", capture_first}, {"last", capture_last}};
                    } else if (operation == "stop_capture") {
                        if (capture_first < 0) throw std::runtime_error("No active block capture");
                        runtime->capture_block(capture_first, capture_last, {});
                        std::ofstream trace(capture_file);
                        if (!trace) throw std::runtime_error("Cannot create block trace");
                        trace << json({{"schema", "questwend-complete-block-v1"},
                            {"model", options.model_path}, {"first", capture_first}, {"last", capture_last},
                            {"context", config.n_ctx}, {"prefill_tokens", capture_prefill_tokens},
                            {"frames", capture_frames}}).dump();
                        if (!trace) throw std::runtime_error("Block trace write failed");
                        response = {{"frames", capture_frames.size()}, {"trace_file", capture_file}};
                        capture_frames = json::array();
                        capture_first = -1;
                    } else if (operation == "restore") {
                        if (snapshot.empty()) throw std::runtime_error("No snapshot");
                        size_t position = 0;
                        runtime->load_probe_state([&](void * bytes, size_t size) {
                            if (size > snapshot.size() - position)
                                throw std::runtime_error("Truncated probe snapshot");
                            std::memcpy(bytes, snapshot.data() + position, size);
                            position += size;
                        });
                        if (position != snapshot.size())
                            throw std::runtime_error("Unconsumed probe snapshot data");
                        pending = saved_pending;
                        response = {{"pending", pending}, {"snapshot_bytes", snapshot.size()}};
                    } else if (operation == "probe") {
                        auto result = runtime->probe_verify_block(
                            request.at("tokens").get<std::vector<int32_t>>(), request.at("keep").get<int>());
                        if (result.kept_tokens) pending = result.next_token;
                        response = verification_json(result);
                    } else if (operation == "verify") {
                        const auto result = runtime->verify_external_greedy(
                            request.value("pending", pending), request.at("drafts").get<std::vector<int32_t>>());
                        pending = result.next_token;
                        response = verification_json(result);
                    } else if (operation == "load_state_file") {
                        const auto filename = request.at("file").get<std::string>();
                        const int restored_pending = request.at("pending").get<int>();
                        if (restored_pending < 0 || restored_pending >= (int) model->vocab().tokens.size())
                            throw std::runtime_error("Invalid persisted pending token");
                        std::ifstream input(filename, std::ios::binary);
                        if (!input) throw std::runtime_error("Cannot open complete probe state");
                        runtime->reset();
                        runtime->load_probe_state([&](void * destination, size_t bytes) {
                            input.read((char *) destination, bytes);
                            if (!input) throw std::runtime_error("Truncated complete probe state");
                        });
                        if (input.peek() != std::char_traits<char>::eof())
                            throw std::runtime_error("Trailing bytes in complete probe state");
                        pending = restored_pending;
                        response = {{"position", runtime->n_past()}, {"pending", pending}, {"file", filename}};
                    } else if (operation == "kv_window") {
                        for (const char * field : {"layer", "offset_bytes", "bytes"}) {
                            if (!request.at(field).is_number_integer() || request.at(field).get<int64_t>() < 0)
                                throw std::runtime_error("KV window numeric fields must be nonnegative integers");
                        }
                        const int64_t layer = request.at("layer").get<int64_t>();
                        if (layer >= (int64_t) model->hparams().n_layer)
                            throw std::runtime_error("KV window layer outside model");
                        const auto window = runtime->probe_kv_window(static_cast<int>(layer),
                            request.at("role").get<std::string>(),
                            request.at("offset_bytes").get<uint64_t>(), request.at("bytes").get<uint64_t>());
                        response = {{"layer", window.layer}, {"role", window.role},
                            {"tensor_name", window.tensor_name}, {"type", "F16"},
                            {"position", window.position}, {"saved_rows", window.saved_rows},
                            {"offset_bytes", window.offset_bytes}, {"bytes", window.words.size() * sizeof(uint16_t)},
                            {"prefix_bytes", window.prefix_bytes}, {"tensor_bytes", window.tensor_bytes},
                            {"shape", window.shape}, {"strides_bytes", window.strides_bytes},
                            {"uint16_words", window.words},
                            {"all_owned_backends_synchronized", window.all_owned_backends_synchronized},
                            {"backends_synchronized", window.backends_synchronized}};
                    } else if (operation == "dump_state") {
                        const std::string filename = request.at("file").get<std::string>();
                        std::ofstream stream(filename, std::ios::binary);
                        if (!stream) throw std::runtime_error("Cannot create state diagnostic export");
                        json chunks = json::array(), layers = json::array();
                        uint64_t offset = 0;
                        runtime->save_probe_state([&](const void * bytes, size_t size) {
                            stream.write((const char *) bytes, size);
                            if (!stream) throw std::runtime_error("State diagnostic export write failed");
                            chunks.push_back({{"offset", offset}, {"bytes", size}});
                            offset += size;
                        });
                        for (int layer = 0; layer < (int) model->hparams().n_layer; ++layer)
                            layers.push_back({{"index", layer},
                                {"recurrent", model->hparams().is_recurrent(layer)},
                                {"kv_columns", model->hparams().head_count_kv(layer) * model->hparams().n_embd_head}});
                        const json manifest = {{"schema", "questwend-probe-state-stream-v1"},
                            {"position", runtime->n_past()}, {"layers", layers}, {"chunks", chunks},
                            {"indexer_columns", model->hparams().indexer_head_dim}};
                        std::ofstream metadata(filename + ".json");
                        metadata << manifest.dump(2);
                        if (!metadata) throw std::runtime_error("State diagnostic manifest write failed");
                        response = {{"bytes", offset}, {"file", filename}, {"manifest", filename + ".json"}};
                    } else if (operation == "dump_qsa_attention") {
                        const auto directory = std::filesystem::path(request.at("directory").get<std::string>());
                        if (std::filesystem::exists(directory))
                            throw std::runtime_error("Refusing an existing attention capture directory");
                        std::filesystem::create_directories(directory);
                        json tensors = json::array();
                        runtime->save_qsa_attention_capture(request.value("layer", 3),
                            [&](const Runtime::QsaCaptureTensor & item, const void * bytes, size_t size) {
                                const auto filename = directory / (item.name + ".bin");
                                std::ofstream stream(filename, std::ios::binary);
                                if (!stream) throw std::runtime_error("Cannot create captured tensor");
                                stream.write((const char *) bytes, size);
                                if (!stream) throw std::runtime_error("Captured tensor write failed");
                                tensors.push_back({{"name", item.name}, {"file", filename.filename().string()},
                                    {"dtype", item.dtype}, {"shape", item.shape}, {"stride_bytes", item.strides},
                                    {"bytes", size}, {"contiguous", item.contiguous},
                                    {"finite_bound", item.finite_bound}, {"scale", item.scale}});
                            });
                        json manifest = {{"schema", "qsa-same-input-attention-capture-v1"},
                            {"layer", 3}, {"position", runtime->n_past()}, {"tensors", tensors},
                            {"same_inputs_within_graph", true}, {"cached_hybrid_graph", true},
                            {"readback_outside_timed_measurement", true},
                            {"contract", "Immediately after one scalar decode; no reset/restore/decode between that step and this export."}};
                        std::ofstream stream(directory / "manifest.json");
                        stream << manifest.dump(2) << '\n';
                        if (!stream) throw std::runtime_error("Attention capture manifest write failed");
                        response = {{"directory", directory.string()}, {"tensors", tensors.size()}};
                    } else if (operation == "profile") {
                        runtime->save_expert_profile(request.at("prefix").get<std::string>());
                        response = {{"saved", true}};
                    } else {
                        throw std::runtime_error("Unknown operation: " + operation);
                    }
                    response["position"] = runtime->n_past();
                }
                response["op"] = operation;
                response["ok"] = true;
                response["wall_ms"] = elapsed_ms(started);
            } catch (const std::exception & error) {
                response = {{"ok", false}, {"error", error.what()}};
            }
            *json_output << response.dump() << std::endl;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "speculation-probe fatal: " << error.what() << std::endl;
        return 1;
    }
}
