#pragma once

// Experimental RPC cache fallback: hashes refer to verified local GGUF ranges.
// No new wire command is required. The coordinator uses SET_TENSOR_HASH.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace indexed_weights {
namespace fs = std::filesystem;
struct Span { fs::path path; uint64_t offset; uint64_t bytes; };

inline bool within(const fs::path & path, const fs::path & root) {
    auto item = path.begin();
    for (auto part = root.begin(); part != root.end(); ++part, ++item)
        if (item == path.end() || *item != *part) return false;
    return item != path.end();
}

inline const std::unordered_map<uint64_t, Span> & index() {
    static std::unordered_map<uint64_t, Span> entries;
    static std::once_flag once;
    std::call_once(once, [] {
        const char * listing = std::getenv("GGML_RPC_MODEL_INDEX");
        const char * directory = std::getenv("GGML_RPC_MODEL_ROOT");
        if (!listing || !directory) return;
        const fs::path root = fs::weakly_canonical(fs::path(directory) / "models");
        std::ifstream input(listing);
        if (!input) throw std::runtime_error("Cannot read local RPC model index");
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            std::string hash_text, offset_text, bytes_text, relative;
            if (!std::getline(row, hash_text, '\t') || !std::getline(row, offset_text, '\t') ||
                !std::getline(row, bytes_text, '\t') || !std::getline(row, relative))
                throw std::runtime_error("Malformed local RPC model index");
            if (!relative.empty() && relative.back() == '\r') relative.pop_back();
            const uint64_t hash = std::stoull(hash_text, nullptr, 16);
            const uint64_t offset = std::stoull(offset_text);
            const uint64_t bytes = std::stoull(bytes_text);
            fs::path source = fs::weakly_canonical(fs::path(directory) / relative);
            if (!within(source, root) || source.extension() != ".gguf" || !bytes || bytes > (UINT64_C(1) << 30))
                throw std::runtime_error("Unsafe source range in local RPC index");
            const uint64_t length = fs::file_size(source);
            if (offset > length || bytes > length - offset)
                throw std::runtime_error("Local RPC index exceeds its model file");
            entries.emplace(hash, Span{source, offset, bytes});
        }
        std::cerr << "local model index ready: " << entries.size() << " weight ranges" << std::endl;
    });
    return entries;
}

inline bool read(uint64_t hash, std::vector<uint8_t> & data) {
    try {
        auto found = index().find(hash);
        if (found == index().end()) return false;
        const Span & source = found->second;
        std::ifstream input(source.path, std::ios::binary);
        input.seekg(static_cast<std::streamoff>(source.offset));
        data.resize(static_cast<size_t>(source.bytes));
        if (!input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()))) return false;
        uint64_t actual = UINT64_C(0xcbf29ce484222325);
        for (uint8_t value : data) { actual ^= value; actual *= UINT64_C(0x100000001b3); }
        if (actual != hash) throw std::runtime_error("Local model bytes differ from requested tensor hash");
        static std::atomic<uint64_t> hits{0}, bytes{0};
        const uint64_t count = ++hits;
        const uint64_t total = bytes.fetch_add(source.bytes) + source.bytes;
        if (count == 1 || count % 1000 == 0)
            std::cerr << "local model index hits: " << count << ", bytes read locally: " << total << std::endl;
        return true;
    } catch (const std::exception & error) {
        std::cerr << "local model index error: " << error.what() << std::endl;
        return false;
    }
}
} // namespace indexed_weights
