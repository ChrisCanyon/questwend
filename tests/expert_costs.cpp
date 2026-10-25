// Isolated, synchronized host-observed expert path calibration.
// Reuses the runtime's ggml backend, routed operations, pinned staging and
// aligned Windows read method. It does not change production model settings.
#include "backend_loader.h"
#include "ggml-cpu.h"
#include "expert_cost_graph.h"
#include <future>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace expert_cost;

static Json read_json(const char * path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open calibration input");
    Json value; input >> value; return value;
}

static ggml_backend_t cuda_backend() {
    auto registry = ggml_backend_reg_by_name("CUDA");
    if (!registry || !ggml_backend_reg_dev_count(registry)) throw std::runtime_error("CUDA device unavailable");
    auto result = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
    if (!result) throw std::runtime_error("CUDA initialization failed");
    return result;
}

static std::vector<size_t> role_offsets(const Expert & shape, int count) {
    return {0, shape[0].bytes * count, (shape[0].bytes + shape[1].bytes) * count};
}

static size_t weight_bytes(const Expert & shape, int count) {
    return (shape[0].bytes + shape[1].bytes + shape[2].bytes) * count;
}

static void copy_selected(const Expert & shape, const HostBuffer & source,
                          const std::vector<size_t> & source_offsets,
                          const std::vector<int> & selected, HostBuffer & destination,
                          const std::vector<size_t> & destination_offsets) {
    for (int role = 0; role < 3; ++role)
        for (size_t index = 0; index < selected.size(); ++index)
            std::memcpy(destination.data + destination_offsets[role] + index * shape[role].bytes,
                source.data + source_offsets[role] + selected[index] * shape[role].bytes, shape[role].bytes);
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: qw-expert-costs corpus.json case.json output.json");
        const Json manifest = read_json(argv[1]), config = read_json(argv[2]);
        const auto & signature = manifest.at("signatures").at(config.at("signature").get<int>());
        const int active = config.at("active_experts"), tokens = config.at("tokens");
        const int repeats = config.value("repeats", 1000), warmup = config.value("warmup", 20);
        const int threads = config.value("cpu_threads", 4), readers = config.value("read_workers", 4);
        const std::string mode = config.at("mode"), path = config.at("path");
        const bool float_reference = mode == "cpu-reference" || mode == "gpu-reference";
        const bool cpu_mode = mode == "cpu" || mode == "cpu-reference";
        const bool gpu_mode = mode == "gpu" || mode == "gpu-reference";
        if (active < 1 || active > 10 || tokens < 1 || tokens > 512 || repeats < 10 || repeats > 10000 ||
            threads < 1 || threads > 16 || readers < 1 || readers > 8 || warmup < 0)
            throw std::runtime_error("Invalid calibration limits");
        std::vector<Expert> corpus;
        for (const auto & item : signature.at("experts")) {
            Expert expert;
            for (const auto & weight : item.at("weights"))
                expert.push_back({weight.at("path").get<std::string>(), weight.at("offset").get<uint64_t>(),
                                  weight.at("bytes").get<size_t>(), weight.at("type").get<int>(),
                                  weight.at("shape")[0].get<int64_t>(), weight.at("shape")[1].get<int64_t>()});
            if (expert.size() != 3) throw std::runtime_error("Incomplete routed expert");
            corpus.push_back(std::move(expert));
        }
        if (corpus.size() < (size_t) active) throw std::runtime_error("Insufficient expert corpus");
        Expert shape = corpus.front();
        for (const auto & expert : corpus)
            for (int role = 0; role < 3; ++role)
                if (expert[role].bytes != shape[role].bytes || expert[role].type != shape[role].type ||
                    expert[role].columns != shape[role].columns || expert[role].rows != shape[role].rows)
                    throw std::runtime_error("Mixed quantization or shape inside a calibration bank");
        const Expert stored_shape = shape;
        if (float_reference) {
            if (path != "resident") throw std::runtime_error("Float reference supports resident graphs only");
            for (auto & slab : shape) {
                slab.type = GGML_TYPE_F32;
                slab.bytes = slab.columns * slab.rows * sizeof(float);
            }
        }

        questwend::initialize_backends();
        const auto initialization = Clock::now();
        ggml_backend_t gpu = cuda_backend();
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) throw std::runtime_error("CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(cpu, threads);
        auto pinned_type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu));
        if (!pinned_type) throw std::runtime_error("CUDA pinned-memory type unavailable");
        const int bank_count = (int) corpus.size();
        const auto bank_offsets = role_offsets(shape, bank_count), offsets = role_offsets(shape, active);
        HostBuffer bank(weight_bytes(shape, bank_count));
        HostBuffer staging(weight_bytes(shape, active));
        HostBuffer pinned(weight_bytes(shape, active), pinned_type);
        HostBuffer evicted(weight_bytes(shape, active));
        HostBuffer eviction_staging(weight_bytes(shape, active), pinned_type);
        ExpertReader buffered(false, readers), direct(true, readers);
        std::vector<int> all(bank_count), compact(active), selected(active);
        std::iota(all.begin(), all.end(), 0); std::iota(compact.begin(), compact.end(), 0);
        const bool disk = path.find("ssd_") == 0;
        const bool is_direct = path.find("direct") != std::string::npos;
        const bool contention = mode == "contention";
        if (float_reference) {
            const auto raw_offsets = role_offsets(stored_shape, bank_count);
            HostBuffer raw_weights(weight_bytes(stored_shape, bank_count));
            buffered.read(corpus, all, raw_offsets, raw_weights.data);
            for (int role = 0; role < 3; ++role) {
                auto traits = ggml_get_type_traits((ggml_type) stored_shape[role].type);
                if (!traits || !traits->to_float) throw std::runtime_error("Missing weight dequantizer for float reference");
                for (int index = 0; index < bank_count; ++index)
                    traits->to_float(raw_weights.data + raw_offsets[role] + index * stored_shape[role].bytes,
                        (float *) (bank.data + bank_offsets[role] + index * shape[role].bytes),
                        shape[role].columns * shape[role].rows);
            }
        } else if (!disk || contention || path == "ssd_buffered") buffered.read(corpus, all, bank_offsets, bank.data);
        std::vector<float> router(active, 1.0f / active);
        if (config.value("routing", std::string("balanced")) == "weight-skew" && active > 1) {
            router.assign(active, .3f / (active - 1)); router.front() = .7f;
        }
        std::unique_ptr<BorrowedCpuWork> cpu_work;
        std::unique_ptr<RoutedGraph> gpu_work;
        if (cpu_mode || contention) cpu_work.reset(new BorrowedCpuWork(cpu, shape, active, tokens, config.value("cpu_copy_control", false)));
        if (gpu_mode || contention)
            gpu_work.reset(new RoutedGraph(gpu, shape, path == "resident" ? bank_count : active, active, tokens));
        if (!cpu_mode && !gpu_mode && !contention) throw std::runtime_error("Unknown case mode");
        if (path == "resident" && gpu_work) gpu_work->upload(shape, bank.data, bank_offsets, bank_count);
        if (!disk) {
            copy_selected(shape, bank, bank_offsets, compact, staging, offsets);
            std::memcpy(pinned.data, staging.data, staging.bytes);
            if (gpu_work && path != "resident") gpu_work->upload(shape, pinned.data, offsets, active);
        }
        const double initialization_ms = elapsed(initialization);
        std::vector<double> observer_times;
        for (int trial = 0; trial < 1000; ++trial) {
            Phases empty;
            const auto start = Clock::now();
            for (const auto * phase : {"read", "upload", "compute", "result"}) measure(empty, phase, [] {});
            observer_times.push_back(elapsed(start));
        }
        std::mt19937 random(config.value("seed", 20261023));
        std::vector<int> selection_pool = all;
        Json raw = Json::array();
        std::map<std::string, std::vector<double>> observations;
        std::vector<float> last_output;
        std::vector<float> parallel_output;
        double first_execution_ms = 0;
        double finite_checksum = 0;
        double maximum_parallel_relative_l2 = 0;
        bool concurrency_equivalence_checked = false;
        bool gpu_weight_bytes_verified = false;

        for (int iteration = -warmup; iteration < repeats; ++iteration) {
            MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
            if (GlobalMemoryStatusEx(&memory) && memory.ullAvailPhys < (2ull << 30))
                throw std::runtime_error("Stopped at 2 GiB available RAM floor");
            if (iteration >= 0 || config.value("rotate_warmup", false)) std::shuffle(selection_pool.begin(), selection_pool.end(), random);
            std::copy_n(selection_pool.begin(), active, selected.begin());
            if (path == "ram_pinned") selected = compact;
            Phases phases;
            const uint64_t bytes_before = buffered.actual_bytes() + direct.actual_bytes();
            const auto started = Clock::now();
            std::future<std::pair<std::vector<float>, Phases>> other;
            if (contention) {
                const auto submitting = Clock::now();
                other = std::async(std::launch::async, [&, submitting] {
                    Phases local;
                    local["queue_delay_ms"] = elapsed(submitting);
                    const auto begin = Clock::now();
                    std::vector<float> output = cpu_work->compute(bank.data, bank_offsets, selected, router, local);
                    local["worker_total_ms"] = elapsed(begin);
                    return std::make_pair(std::move(output), local);
                });
                phases["cpu_submit_ms"] = elapsed(submitting);
            }
            uint8_t * payload = bank.data;
            auto payload_offsets = bank_offsets;
            auto payload_ids = selected;
            if (disk) {
                measure(phases, "source_read_ms", [&] {
                    (is_direct ? direct : buffered).read(corpus, selected, offsets,
                        cpu_mode ? staging.data : pinned.data);
                });
                payload = cpu_mode ? staging.data : pinned.data;
                payload_offsets = offsets; payload_ids = compact;
            } else if (path != "resident" && path != "dispatch") {
                if (path == "evict_reload" && gpu_work) {
                    measure(phases, "gpu_eviction_download_ms", [&] {
                        gpu_work->download(shape, eviction_staging.data, offsets, active);
                    });
                    measure(phases, "ram_eviction_copy_ms", [&] {
                        std::memcpy(evicted.data, eviction_staging.data, evicted.bytes);
                    });
                }
                if (path == "ram_pinned") {
                    // Best-case pinned source, already populated before the
                    // timer. Keep separate from real pageable-cache staging.
                    payload = pinned.data; payload_offsets = offsets; payload_ids = compact;
                } else {
                    HostBuffer & destination = cpu_mode || path == "ram_pageable" ? staging : pinned;
                    measure(phases, "ram_staging_copy_ms", [&] {
                        copy_selected(shape, bank, bank_offsets, selected, destination, offsets);
                    });
                    payload = destination.data; payload_offsets = offsets; payload_ids = compact;
                }
            }
            if (cpu_mode) {
                if (path == "dispatch" || path == "ssd_direct_dispatch") {
                    const auto submitting = Clock::now();
                    auto future = std::async(std::launch::async, [&, submitting] {
                        Phases local; local["cpu_queue_delay_ms"] = elapsed(submitting);
                        const auto worker_start = Clock::now();
                        std::vector<float> output = cpu_work->compute(payload, payload_offsets, payload_ids, router, local);
                        local["cpu_worker_total_ms"] = elapsed(worker_start);
                        return std::make_pair(std::move(output), local);
                    });
                    phases["cpu_submit_ms"] = elapsed(submitting);
                    auto completed = future.get();
                    last_output = std::move(completed.first);
                    for (auto & item : completed.second) phases[item.first] += item.second;
                } else last_output = cpu_work->compute(payload, payload_offsets, payload_ids, router, phases);
            } else {
                if (path != "resident") measure(phases, "gpu_weight_upload_ms", [&] {
                    gpu_work->upload(shape, payload, payload_offsets, active);
                });
                last_output = gpu_work->compute(path == "resident" ? selected : compact, router, phases);
            }
            if (contention) {
                const auto joining = Clock::now();
                auto completed = other.get(); phases["cpu_join_ms"] = elapsed(joining);
                for (auto & item : completed.second) phases["parallel_cpu_" + item.first] += item.second;
                parallel_output = std::move(completed.first);
            }
            phases["total_ms"] = elapsed(started);
            if (gpu_work && !gpu_weight_bytes_verified) {
                const int count = path == "resident" ? bank_count : active;
                const auto checked_offsets = role_offsets(shape, count);
                HostBuffer readback(weight_bytes(shape, count));
                gpu_work->download(shape, readback.data, checked_offsets, count);
                const uint8_t * expected = path == "resident" ? bank.data : payload;
                const auto expected_offsets = path == "resident" ? bank_offsets : payload_offsets;
                for (int role = 0; role < 3; ++role)
                    if (std::memcmp(readback.data + checked_offsets[role], expected + expected_offsets[role],
                                    count * shape[role].bytes) != 0)
                        throw std::runtime_error("GPU expert upload/readback changed weight bytes");
                gpu_weight_bytes_verified = true;
            }
            if (iteration == -warmup) first_execution_ms = phases["total_ms"];
            if (last_output.empty()) throw std::runtime_error("No expert output");
            for (float value : last_output) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite expert output");
            if (contention) {
                if (parallel_output.size() != last_output.size()) throw std::runtime_error("Concurrent output sizes differ");
                double difference = 0, reference = 0;
                for (size_t column = 0; column < last_output.size(); ++column) {
                    if (!std::isfinite(parallel_output[column])) throw std::runtime_error("Nonfinite concurrent CPU output");
                    const double delta = parallel_output[column] - last_output[column];
                    difference += delta * delta;
                    reference += (double) parallel_output[column] * parallel_output[column];
                }
                const double relative = std::sqrt(difference / std::max(reference, 1e-24));
                maximum_parallel_relative_l2 = std::max(maximum_parallel_relative_l2, relative);
                if (!concurrency_equivalence_checked) {
                    // Compare each backend with itself. CPU/GPU activation
                    // quantization differs, so their pairwise difference alone
                    // is not a concurrency correctness test.
                    Phases ignored;
                    const auto cpu_sequential = cpu_work->compute(bank.data, bank_offsets, selected, router, ignored);
                    const auto gpu_sequential = gpu_work->compute(compact, router, ignored);
                    auto same = [](const std::vector<float> & first, const std::vector<float> & second) {
                        double error = 0, norm = 0;
                        for (size_t index = 0; index < first.size(); ++index) {
                            const double delta = first[index] - second[index];
                            error += delta * delta; norm += (double) first[index] * first[index];
                        }
                        return std::sqrt(error / std::max(norm, 1e-24)) <= 1e-6;
                    };
                    if (!same(cpu_sequential, parallel_output) || !same(gpu_sequential, last_output))
                        throw std::runtime_error("Concurrent execution changed a backend's output");
                    concurrency_equivalence_checked = true;
                }
            }
            finite_checksum += last_output.front();
            if (iteration >= 0) {
                for (auto & item : phases) observations[item.first].push_back(item.second);
                raw.push_back({{"trial", iteration}, {"selected", selected}, {"phases", phases},
                    {"file_read_bytes", buffered.actual_bytes() + direct.actual_bytes() - bytes_before}});
            }
        }
        Json summaries;
        for (auto & item : observations) summaries[item.first] = distribution(item.second);
        const std::string destination = argv[3];
        Json report{{"state", "done"}, {"config", config}, {"backend", cpu_mode ? "CPU" : "CUDA0"},
            {"weight_bytes_per_selected_group", weight_bytes(shape, active)}, {"corpus_experts", bank_count},
            {"corpus_bytes", bank.bytes}, {"backend_and_graph_initialization_ms", initialization_ms},
            {"cpu_graph_allocation_ms", cpu_work ? cpu_work->allocation_ms : 0},
            {"gpu_graph_allocation_ms", gpu_work ? gpu_work->allocation_ms : 0},
            {"first_execution_ms", first_execution_ms}, {"distributions", summaries}, {"raw_trials", raw},
            {"four_empty_phase_observer_cost", distribution(observer_times)},
            {"logical_read_bytes_per_trial", disk ? weight_bytes(shape, active) : 0},
            {"all_read_bytes_including_setup_and_warmup", buffered.actual_bytes() + direct.actual_bytes()},
            {"cpu_batch_above_one_is_proposed_prefill_path", mode != "gpu" && tokens > 1},
            {"timing_scope", "Synchronized host wall time, including submission, activation copies, graph execution and result access. Phase timers overlap in concurrent cases."},
            {"cold_state", is_direct ? "Random reads bypass Windows file cache; SSD hardware cache is uncontrolled" : disk ? "Buffered randomized reads; file-cache state is warm after repetition" : "Resident memory; expert identities rotate except the fixed pinned-source control"},
            {"output_checksum", finite_checksum},
            {"concurrent_maximum_relative_l2", maximum_parallel_relative_l2},
            {"concurrency_same_backend_equivalence_checked", concurrency_equivalence_checked},
            {"gpu_weight_bytes_verified", gpu_weight_bytes_verified},
            {"validation_outside_trial_timer", true}};
        std::ofstream binary(destination + ".output.f32", std::ios::binary);
        binary.write((const char *) last_output.data(), last_output.size() * sizeof(float));
        std::ofstream saved(destination); saved << report.dump(2); saved.close();
        report.erase("raw_trials"); std::cout << report.dump() << std::endl;
        cpu_work.reset(); gpu_work.reset();
        ggml_backend_free(cpu); ggml_backend_free(gpu);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl; return 1;
    }
}
