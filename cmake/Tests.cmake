# Regression fixtures are separate executables, never server sources.
function(qw_native_test name)
    string(REPLACE "_" "-" suffix "${name}")
    set(target "qw-${suffix}-test")
    add_executable(${target} "tests/${name}_test.cpp")
    target_include_directories(${target} PRIVATE core tests third_party/ggml/src)
    target_link_libraries(${target} PRIVATE ggml ggml-cpu Threads::Threads)
    if(WIN32)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    if(name STREQUAL "expert_demand_pipeline" OR name STREQUAL "host_cache_s3fifo")
        target_compile_definitions(${target} PRIVATE QWEN_CACHE_TEST_FAULTS)
    endif()
    if(name STREQUAL "token_pieces")
        target_sources(${target} PRIVATE core/token_pieces.cpp)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endfunction()
foreach(name
    token_pieces
    cache_memory_budget
    direct_pinned_covering
    expert_demand_pipeline
    expert_demand_trace
    expert_install_check
    expert_io_targeted_wake
    expert_io_workers
    expert_ssd_read_phase
    expert_weight_audit
    gpu_lru_index
    host_cache
    host_cache_s3fifo
    prefill_cache_policy
    prompt_checkpoint_memory
    qsa_constant_inputs
    qsa_constant_storage
    qsa_identity_view
    ram_ssd_admission
    tiny_batch_workspace
    workspace_partition)
    qw_native_test(${name})
endforeach()
