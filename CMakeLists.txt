cmake_minimum_required(VERSION 3.20)
project(questwend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
option(BUILD_TESTING "Build native regression tests" OFF)
option(QW_BUILD_DIAGNOSTICS "Build standalone calibration and GPU probes" OFF)
set(QW_CHAT_ADAPTER_DIR "" CACHE PATH "Pinned upstream chat adapter directory")
if(NOT EXISTS "${QW_CHAT_ADAPTER_DIR}/questwend_bridge.h")
    message(FATAL_ERROR "The production server requires the pinned upstream chat adapter")
endif()
if(GGML_RPC)
    message(FATAL_ERROR "Private RPC is retired from the production core")
endif()
set(GGML_RPC OFF CACHE BOOL "" FORCE)
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


# Optional matching backend SDK permits host-only refactoring validation.
set(QW_BACKEND_BUILD "" CACHE PATH "Optional matching ggml build SDK")
if(QW_BACKEND_BUILD)
    include(cmake/ImportedBackend.cmake)
else()
    add_subdirectory(third_party/ggml)
endif()
find_package(Threads REQUIRED)
add_library(questwend_core STATIC
    core/args.cpp core/gguf_util.cpp core/model.cpp core/token_pieces.cpp
    core/runtime.cpp core/expert_cache.cpp core/expert_debug.cpp core/cold_cache_prime.cpp core/ngram_table.cpp core/sampler.cpp)
target_include_directories(questwend_core PUBLIC core)
target_include_directories(questwend_core PRIVATE third_party/ggml/src)
target_link_libraries(questwend_core PUBLIC ggml Threads::Threads)
add_executable(qw-server server/main.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(qw-server PRIVATE third_party/cpp-httplib third_party/nlohmann "${QW_CHAT_ADAPTER_DIR}")
target_link_libraries(qw-server PRIVATE questwend_core Threads::Threads)
if(WIN32)
    target_link_libraries(qw-server PRIVATE ws2_32)
    add_custom_command(TARGET qw-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:qw-server> $<TARGET_FILE_DIR:qw-server>
        COMMAND_EXPAND_LISTS)
endif()
include(CTest)
if(BUILD_TESTING)
    include(cmake/Tests.cmake)
endif()
if(QW_BUILD_DIAGNOSTICS)
    include(cmake/Diagnostics.cmake)
endif()
