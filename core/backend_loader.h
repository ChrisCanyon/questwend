#pragma once

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace questwend {

// The released b005 CUDA DLL exports its registry factory, rather than the
// optional plugin entry point. This explicit experimental path reuses that
// matching CUDA build while building the CPU/RPC engine without a CUDA toolkit.
inline void initialize_backends() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_backend_load_all();
#ifdef _WIN32
        const char * path = std::getenv("QW_CUDA_BACKEND");
        if (path && *path && !ggml_backend_reg_by_name("CUDA")) {
            HMODULE library = LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!library) throw std::runtime_error("Cannot load the explicitly configured CUDA backend DLL");
            using Factory = ggml_backend_reg_t (*)();
            auto factory = reinterpret_cast<Factory>(GetProcAddress(library, "ggml_backend_cuda_reg"));
            if (!factory) throw std::runtime_error("CUDA backend registry factory is unavailable");
            auto registry = factory();
            if (!registry || registry->api_version != GGML_BACKEND_API_VERSION)
                throw std::runtime_error("CUDA backend API version differs from the isolated runtime");
            ggml_backend_register(registry);
            // Registry interfaces retain pointers into the DLL for process life.
        }
#endif
    });
}

} // namespace questwend
