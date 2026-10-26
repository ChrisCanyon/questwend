cmake_minimum_required(VERSION 3.20)
project(questwend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- options ----
option(QW_CUDA   "Enable CUDA backend"            OFF)
option(QW_HIP    "Enable HIP (AMD ROCm) backend"  OFF)
option(QW_METAL  "Enable Metal backend"           OFF)
option(QW_VULKAN "Enable Vulkan backend"          OFF)

if(QW_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # ggml defaults CUDA graphs OFF for standalone builds ("llama.cpp only");
    # enable them to eliminate per-kernel launch overhead during decode.
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()
if(QW_HIP)
    # AMD. Pick the arch with -DGPU_TARGETS=gfx#### (see `rocminfo | grep gfx`);
    # ggml forwards it to CMAKE_HIP_ARCHITECTURES. Unlike GGML_CUDA_GRAPHS,
    # ggml already defaults GGML_HIP_GRAPHS to ON, so there is nothing to force.
    # NOTE: untested on real hardware -- the engine itself is backend-agnostic
    # (it takes the first GPU device ggml reports, discrete before integrated),
    # but the expert-offload paths have only ever been exercised on CUDA and Metal.
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()
if(QW_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(QW_VULKAN)
    # Vendor-neutral GPU path (AMD, Intel, NVIDIA), and the way to use a Ryzen
    # or Intel integrated GPU without installing ROCm or oneAPI. Needs the
    # Vulkan SDK: ggml compiles its shaders with the glslc that ships with it.
    # NOTE: untested on real hardware, same caveat as QW_HIP above.
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

# ---- vendored ggml (self-contained, see third_party/ggml/GGML_VERSION) ----
add_subdirectory(third_party/ggml)

# ---- core engine library ----
add_library(questwend_core STATIC
    core/args.cpp
    core/gguf_util.cpp
    core/model.cpp
    core/tokenizer.cpp
    core/runtime.cpp
    core/expert_cache.cpp
    core/ngram_table.cpp
    core/sampler.cpp
    core/chat.cpp
    core/vision.cpp
)
target_include_directories(questwend_core PUBLIC core)
target_include_directories(questwend_core PRIVATE third_party/stb third_party/ggml/src)
target_link_libraries(questwend_core PUBLIC ggml)

# Small native correctness gate for the bounded per-node RAM cache.
add_executable(qw-host-cache-test tests/host_cache_test.cpp)
target_include_directories(qw-host-cache-test PRIVATE core)

add_executable(qw-expert-calibrate tests/expert_calibrate.cpp)
target_include_directories(qw-expert-calibrate PRIVATE core third_party/nlohmann third_party/ggml/src)
target_link_libraries(qw-expert-calibrate PRIVATE ggml)

# Local expert load/compute path distributions; no model-server changes.
add_executable(qw-expert-costs tests/expert_costs.cpp)
target_include_directories(qw-expert-costs PRIVATE core tests third_party/nlohmann third_party/ggml/src)
target_link_libraries(qw-expert-costs PRIVATE ggml)

# ---- CLI ----
add_executable(qw-cli cli/main.cpp)
target_link_libraries(qw-cli PRIVATE questwend_core)

# ---- tokenizer cross-check tool (tests/tokenize_main.cpp) ----
add_executable(qw-tokenize tests/tokenize_main.cpp)
target_link_libraries(qw-tokenize PRIVATE questwend_core)

# ---- Server (OpenAI-compatible) ----
add_executable(qw-tool-call-probe EXCLUDE_FROM_ALL tests/tool_call_probe.cpp)
target_include_directories(qw-tool-call-probe PRIVATE server third_party/nlohmann)

add_executable(qw-chat-template-probe EXCLUDE_FROM_ALL tests/chat_template_probe.cpp)
target_include_directories(qw-chat-template-probe PRIVATE third_party/nlohmann)
target_link_libraries(qw-chat-template-probe PRIVATE questwend_core)

add_executable(qw-server server/main.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(qw-server PRIVATE
    third_party/cpp-httplib
    third_party/nlohmann
)
target_link_libraries(qw-server PRIVATE questwend_core)
# The helper is a separate process: never link two ggml ABIs into the server.
set(QW_CHAT_ADAPTER_DIR "" CACHE PATH "Optional external upstream chat adapter directory")
if(QW_CHAT_ADAPTER_DIR)
    target_include_directories(qw-server PRIVATE "${QW_CHAT_ADAPTER_DIR}")
    target_compile_definitions(qw-server PRIVATE QW_UPSTREAM_CHAT=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(qw-server PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(qw-server PRIVATE ws2_32)
endif()

# ---- keep the ggml DLLs next to the executables (Windows) ----
# ggml builds its DLLs into build/bin/<config> while the executables land in
# build/<config>, so Windows resolves them from the exe directory instead --
# a stale copy there silently shadows a freshly built ggml. Copy them after
# every link so the two can never drift apart.
if(WIN32)
    foreach(tgt qw-cli qw-server qw-tokenize)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${tgt}> $<TARGET_FILE_DIR:${tgt}>
            COMMAND_EXPAND_LISTS)
    endforeach()
endif()

# Experimental loopback worker, matched to this vendored ggml protocol.
if(GGML_RPC)
    add_executable(qw-rpc-server server/rpc_server.cpp)
    target_link_libraries(qw-rpc-server PRIVATE ggml)
    target_include_directories(qw-rpc-server PRIVATE core third_party/ggml/src)
endif()

if(GGML_RPC)
    add_executable(qw-rpc-probe tests/rpc_probe.cpp)
    target_link_libraries(qw-rpc-probe PRIVATE ggml)
endif()

# Apply after the existing questwend_core target has been defined. This target
# must link the isolated core with block-runtime.patch; never production core.
add_executable(qw-block-worker tests/block_worker.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(qw-block-worker PRIVATE core third_party/cpp-httplib third_party/nlohmann)
target_link_libraries(qw-block-worker PRIVATE questwend_core Threads::Threads)
if(WIN32)
    target_link_libraries(qw-block-worker PRIVATE ws2_32)
endif()

# Isolated external-draft feasibility and correctness probe.
add_executable(qw-speculation-probe tests/speculation_probe.cpp)
target_include_directories(qw-speculation-probe PRIVATE third_party/nlohmann)
target_link_libraries(qw-speculation-probe PRIVATE questwend_core)

# Binary complete-layer relay in the same isolated speculation process.
target_sources(qw-speculation-probe PRIVATE third_party/cpp-httplib/httplib.cpp)
target_include_directories(qw-speculation-probe PRIVATE third_party/cpp-httplib)
target_link_libraries(qw-speculation-probe PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(qw-speculation-probe PRIVATE ws2_32)
endif()

# Add only to the coordinated isolated QuestWend build, after core exists.
add_executable(qw-real-grouped-probe tests/real_grouped_probe.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(qw-real-grouped-probe PRIVATE core tests third_party/cpp-httplib third_party/nlohmann third_party/ggml/src)
target_link_libraries(qw-real-grouped-probe PRIVATE questwend_core Threads::Threads)
if(WIN32)
    target_link_libraries(qw-real-grouped-probe PRIVATE ws2_32)
    target_compile_definitions(qw-real-grouped-probe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# No model/core dependency: only the exact same bundled HTTP implementation.
add_executable(qw-block-transport-echo tests/block_transport_echo.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(qw-block-transport-echo PRIVATE third_party/cpp-httplib third_party/nlohmann)
target_link_libraries(qw-block-transport-echo PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(qw-block-transport-echo PRIVATE ws2_32)
    target_compile_definitions(qw-block-transport-echo PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Explicit synthetic target: no model, CUDA backend or GGUF needed.
add_executable(qw-expert-io-workers-test EXCLUDE_FROM_ALL tests/expert_io_workers_test.cpp)
target_include_directories(qw-expert-io-workers-test PRIVATE core)
target_compile_features(qw-expert-io-workers-test PRIVATE cxx_std_17)
target_link_libraries(qw-expert-io-workers-test PRIVATE Threads::Threads)
if(WIN32)
    target_compile_definitions(qw-expert-io-workers-test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_executable(qw-workspace-partition-test EXCLUDE_FROM_ALL tests/workspace_partition_test.cpp)
target_include_directories(qw-workspace-partition-test PRIVATE core)
target_compile_features(qw-workspace-partition-test PRIVATE cxx_std_17)

add_executable(qw-prefill-cache-policy-test EXCLUDE_FROM_ALL tests/prefill_cache_policy_test.cpp)
target_include_directories(qw-prefill-cache-policy-test PRIVATE core)
target_compile_features(qw-prefill-cache-policy-test PRIVATE cxx_std_17)

add_executable(qw-cache-memory-budget-test EXCLUDE_FROM_ALL tests/cache_memory_budget_test.cpp)
target_include_directories(qw-cache-memory-budget-test PRIVATE core)
target_compile_features(qw-cache-memory-budget-test PRIVATE cxx_std_17)

# Pure host input/lifetime test: no model or CUDA dependency.
add_executable(qw-qsa-constant-inputs-test EXCLUDE_FROM_ALL tests/qsa_constant_inputs_test.cpp)
target_include_directories(qw-qsa-constant-inputs-test PRIVATE core)
target_compile_features(qw-qsa-constant-inputs-test PRIVATE cxx_std_17)

# Actual allocator/CPU replay gate: verifies the INPUT|OUTPUT lifetime contract.
add_executable(qw-qsa-constant-storage-test EXCLUDE_FROM_ALL tests/qsa_constant_storage_test.cpp)
target_include_directories(qw-qsa-constant-storage-test PRIVATE core)
target_link_libraries(qw-qsa-constant-storage-test PRIVATE ggml ggml-cpu)
target_compile_features(qw-qsa-constant-storage-test PRIVATE cxx_std_17)

# Pure policy test: no model weights or GPU work.
add_executable(qw-ram-ssd-admission-test EXCLUDE_FROM_ALL tests/ram_ssd_admission_test.cpp)
target_include_directories(qw-ram-ssd-admission-test PRIVATE core)
target_compile_features(qw-ram-ssd-admission-test PRIVATE cxx_std_17)

# Pure metadata gates; no model/GPU required.
add_executable(qw-tiny-batch-workspace-test EXCLUDE_FROM_ALL tests/tiny_batch_workspace_test.cpp)

# Optional CPU-only ownership/readiness unit; no model, CUDA or weights.
add_executable(qw-expert-demand-pipeline-test EXCLUDE_FROM_ALL tests/expert_demand_pipeline_test.cpp)
target_include_directories(qw-expert-demand-pipeline-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_compile_features(qw-expert-demand-pipeline-test PRIVATE cxx_std_17)
target_link_libraries(qw-expert-demand-pipeline-test PRIVATE Threads::Threads)
target_compile_definitions(qw-expert-demand-pipeline-test PRIVATE QWEN_CACHE_TEST_FAULTS)
if (WIN32)
    target_compile_definitions(qw-expert-demand-pipeline-test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_executable(qw-expert-weight-audit-test EXCLUDE_FROM_ALL tests/expert_weight_audit_test.cpp)
target_include_directories(qw-expert-weight-audit-test PRIVATE core)
target_compile_features(qw-expert-weight-audit-test PRIVATE cxx_std_17)
if (WIN32)
    target_compile_definitions(qw-expert-weight-audit-test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Opt-in experimental QSA numerical diagnostic; never runs during a build.
add_executable(qw-qsa-attention-reference tests/qsa_attention_reference.cpp)
target_include_directories(qw-qsa-attention-reference PRIVATE
    core third_party/ggml/src third_party/nlohmann)
target_link_libraries(qw-qsa-attention-reference PRIVATE ggml)

# CPU-only framing/capacity/no-hot-allocation checks; no model or GPU.
add_executable(qw-expert-demand-trace-test EXCLUDE_FROM_ALL tests/expert_demand_trace_test.cpp)
target_include_directories(qw-expert-demand-trace-test PRIVATE core)
target_compile_features(qw-expert-demand-trace-test PRIVATE cxx_std_17)
if (WIN32)
    target_compile_definitions(qw-expert-demand-trace-test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Model-free exact scalar gather CPU-reference gate; GPU launch only when run.
add_executable(qw-scalar-get-rows-probe EXCLUDE_FROM_ALL tests/scalar_get_rows_probe.cpp)
target_include_directories(qw-scalar-get-rows-probe PRIVATE core third_party/ggml/src)
target_link_libraries(qw-scalar-get-rows-probe PRIVATE ggml)
target_compile_features(qw-scalar-get-rows-probe PRIVATE cxx_std_17)

# Model-free exact vector gather CPU-reference gate. GPU work only when run.
add_executable(qw-vector-get-rows-probe EXCLUDE_FROM_ALL tests/vector_get_rows_probe.cpp)
target_include_directories(qw-vector-get-rows-probe PRIVATE core third_party/ggml/src)
target_link_libraries(qw-vector-get-rows-probe PRIVATE ggml)
target_compile_features(qw-vector-get-rows-probe PRIVATE cxx_std_17)

# CPU-only strict prompt-checkpoint limit and retained-payload accounting gate.
add_executable(qw-prompt-checkpoint-memory-test EXCLUDE_FROM_ALL tests/prompt_checkpoint_memory_test.cpp)
target_include_directories(qw-prompt-checkpoint-memory-test PRIVATE core)
target_compile_features(qw-prompt-checkpoint-memory-test PRIVATE cxx_std_17)

# Synthetic CPU-only targeted wake / generation / exception correctness gate.
add_executable(qw-expert-io-targeted-wake-test EXCLUDE_FROM_ALL tests/expert_io_targeted_wake_test.cpp)
target_include_directories(qw-expert-io-targeted-wake-test PRIVATE core)
target_compile_features(qw-expert-io-targeted-wake-test PRIVATE cxx_std_17)
target_link_libraries(qw-expert-io-targeted-wake-test PRIVATE Threads::Threads)

# Exact GPU LRU metadata/mutation-hook equivalence; CPU-only and model-free.
add_executable(qw-gpu-lru-index-test EXCLUDE_FROM_ALL tests/gpu_lru_index_test.cpp)
target_include_directories(qw-gpu-lru-index-test PRIVATE core)
target_compile_features(qw-gpu-lru-index-test PRIVATE cxx_std_17)

# Actual HostExpertCache S3FIFO ownership/publication gate; CPU-only.
add_executable(qw-host-cache-s3fifo-test EXCLUDE_FROM_ALL tests/host_cache_s3fifo_test.cpp)
target_include_directories(qw-host-cache-s3fifo-test PRIVATE core)
target_compile_features(qw-host-cache-s3fifo-test PRIVATE cxx_std_17)
target_compile_definitions(qw-host-cache-s3fifo-test PRIVATE QWEN_CACHE_TEST_FAULTS)
target_link_libraries(qw-host-cache-s3fifo-test PRIVATE Threads::Threads)

# CPU-only SSD phase and unchanged direct covering-read correctness gate.
add_executable(qw-expert-ssd-phase-test EXCLUDE_FROM_ALL tests/expert_ssd_read_phase_test.cpp)
target_include_directories(qw-expert-ssd-phase-test PRIVATE core)
target_compile_features(qw-expert-ssd-phase-test PRIVATE cxx_std_17)
target_link_libraries(qw-expert-ssd-phase-test PRIVATE Threads::Threads)
if(WIN32)
    target_compile_definitions(qw-expert-ssd-phase-test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# CPU-only covered-role predicate fixture; no model/backend initialization.
add_executable(qw-expert-install-check-test EXCLUDE_FROM_ALL tests/expert_install_check_test.cpp)
target_include_directories(qw-expert-install-check-test PRIVATE core)
target_compile_features(qw-expert-install-check-test PRIVATE cxx_std_17)

# Model-free CPU identity-prefix graph, named leaf and write/replay gate.
add_executable(qw-qsa-identity-view-test EXCLUDE_FROM_ALL tests/qsa_identity_view_test.cpp)
target_link_libraries(qw-qsa-identity-view-test PRIVATE ggml ggml-cpu)
target_compile_features(qw-qsa-identity-view-test PRIVATE cxx_std_17)

# Model-free matched-CUDA exact QSA pooling byte gate; never a service.
add_executable(qw-qsa-pool4-probe EXCLUDE_FROM_ALL tests/qsa_pool4_probe.cpp)
target_include_directories(qw-qsa-pool4-probe PRIVATE core third_party/ggml/src)
target_link_libraries(qw-qsa-pool4-probe PRIVATE ggml)
target_compile_features(qw-qsa-pool4-probe PRIVATE cxx_std_17)

# CPU-only Windows covering-range file/sentinel fixture; no CUDA/model init.
if(WIN32)
    add_executable(qw-direct-pinned-covering-test EXCLUDE_FROM_ALL tests/direct_pinned_covering_test.cpp)
    target_include_directories(qw-direct-pinned-covering-test PRIVATE core)
    target_compile_features(qw-direct-pinned-covering-test PRIVATE cxx_std_17)
endif()
