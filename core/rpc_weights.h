#pragma once

// Map a source-file range to its precomputed weight identifier. Remote pools
// request that identifier; only the assigned node reads the GGUF weight bytes.
#include "ggml-backend.h"
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace questwend {
// Canonicalize each distinct source path once, rather than once per expert.
inline std::string canonical_weight_path(const std::string & source) {
    static std::mutex mutex;
    static std::map<std::string, std::string> paths;
    std::lock_guard<std::mutex> guard(mutex);
    auto found = paths.find(source);
    if (found == paths.end()) found = paths.emplace(source, std::filesystem::weakly_canonical(source).string()).first;
    return found->second;
}

inline uint64_t remote_expert_hash(const std::string & source, uint64_t offset, size_t bytes) {
    const char * listing = std::getenv("QW_RPC_WEIGHT_INDEX");
    const char * root = std::getenv("QW_RPC_MODEL_ROOT");
    if (!listing || !root) throw std::runtime_error("Remote expert pools require a local source-range index");
    using Key = std::tuple<std::string, uint64_t, uint64_t>;
    static std::map<Key, uint64_t> hashes;
    static std::once_flag once;
    std::call_once(once, [&] {
        std::ifstream input(listing);
        if (!input) throw std::runtime_error("Cannot open coordinator weight index");
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            std::string hash, start, length, relative;
            if (!std::getline(row, hash, '\t') || !std::getline(row, start, '\t') ||
                !std::getline(row, length, '\t') || !std::getline(row, relative))
                throw std::runtime_error("Malformed coordinator weight index");
            if (!relative.empty() && relative.back() == '\r') relative.pop_back();
            const auto path = canonical_weight_path((std::filesystem::path(root) / relative).string());
            hashes.emplace(Key{path, std::stoull(start), std::stoull(length)}, std::stoull(hash, nullptr, 16));
        }
    });
    auto found = hashes.find(Key{canonical_weight_path(source), offset, bytes});
    if (found == hashes.end()) throw std::runtime_error("Remote expert range missing from coordinator index");
    return found->second;
}

inline bool fill_remote_expert(ggml_backend_t backend, ggml_tensor * destination,
                               const std::string & source, uint64_t offset,
                               size_t bytes, size_t destination_offset) {
    if (std::string(ggml_backend_name(backend)).rfind("RPC", 0) != 0) return false;
    const auto hash = remote_expert_hash(source, offset, bytes);
    auto registry = ggml_backend_reg_by_name("RPC");
    using Fill = bool (*)(ggml_tensor *, uint64_t, size_t, size_t);
    auto fill = reinterpret_cast<Fill>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_rpc_tensor_set_hash"));
    if (!fill || !fill(destination, hash, destination_offset, bytes))
        throw std::runtime_error("Worker could not load the assigned local expert range");
    return true;
}
inline void fill_remote_experts(std::vector<ggml_tensor *> & destinations, const std::vector<uint64_t> & hashes,
                                const std::vector<size_t> & offsets, const std::vector<size_t> & sizes) {
    if (destinations.empty()) return;
    if (destinations.size() != hashes.size() || offsets.size() != hashes.size() || sizes.size() != hashes.size())
        throw std::runtime_error("Remote identifier batch sizes differ");
    auto registry = ggml_backend_reg_by_name("RPC");
    using Fill = bool (*)(ggml_tensor **, const uint64_t *, const size_t *, const size_t *, size_t);
    auto fill = reinterpret_cast<Fill>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_rpc_tensors_set_hash"));
    if (!fill || !fill(destinations.data(), hashes.data(), offsets.data(), sizes.data(), hashes.size()))
        throw std::runtime_error("Worker could not load the assigned expert batch");
}
} // namespace questwend
