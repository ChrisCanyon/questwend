#include "expert_cache.h"
#include "expert_io_workers.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace questwend {

// Experimental startup policy: fill empty managed tiers before the first prompt.
// Normal source-prefill and decode code paths are unchanged. A full retained
// GPU pool sends new source-prefill misses to the existing transient workspace;
// decode can later evict these initial placements using its usual policy.
void ExpertCache::prime_cold() {
    if (!ssd_ || !host_cache_ || readthrough_ram_ ||
        std::string(ggml_backend_name(backend_)).rfind("RPC", 0) == 0)
        throw std::runtime_error("Cold cache priming requires local SSD-tier GPU and exclusive RAM cache");
    const auto start = std::chrono::steady_clock::now();
    const char * selected_layout = std::getenv("QWEN_COLD_PRIME_LAYOUT");
    const std::string layout = selected_layout ? selected_layout : "sequential";
    if (layout != "sequential" && layout != "balanced-rotated" && layout != "learned")
        throw std::invalid_argument("QWEN_COLD_PRIME_LAYOUT must be sequential, balanced-rotated or learned");

    struct Candidate { int layer; int expert; };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(n_layer_) * n_expert_);
    if (layout == "learned") {
        const char * order_path = std::getenv("QWEN_COLD_PRIME_ORDER");
        if (!order_path || !*order_path)
            throw std::invalid_argument("Learned cold priming needs QWEN_COLD_PRIME_ORDER");
        std::ifstream input(order_path);
        std::string magic;
        int version = 0, layers = 0, experts = 0;
        if (!(input >> magic >> version >> layers >> experts) ||
            magic != "QWCOLDORDER" || version != 1 ||
            layers != n_layer_ || experts != n_expert_)
            throw std::runtime_error("Cold-prime order header does not match the model");
        std::vector<std::vector<int>> order(n_layer_, std::vector<int>(n_expert_));
        for (int layer = 0; layer < n_layer_; ++layer) {
            std::vector<bool> seen(n_expert_, false);
            for (int rank = 0; rank < n_expert_; ++rank) {
                int expert = -1;
                if (!(input >> expert) || expert < 0 || expert >= n_expert_ || seen[expert])
                    throw std::runtime_error("Cold-prime order has a missing or duplicate expert");
                order[layer][rank] = expert;
                seen[expert] = true;
            }
        }
        std::string extra;
        if (input >> extra)
            throw std::runtime_error("Cold-prime order contains extra entries");
        for (int rank = 0; rank < n_expert_; ++rank)
            for (int layer = 0; layer < n_layer_; ++layer)
                if (serves(layer)) candidates.push_back({layer, order[layer][rank]});
    } else if (layout == "balanced-rotated") {
        // Sweep the same rank in every layer before taking another expert.
        // A per-layer rotation avoids always selecting the lowest expert IDs.
        for (int rank = 0; rank < n_expert_; ++rank)
            for (int layer = 0; layer < n_layer_; ++layer)
                if (serves(layer))
                    candidates.push_back({layer, (rank + layer * 131 + 17) % n_expert_});
    } else {
        for (int layer = 0; layer < n_layer_; ++layer)
            if (serves(layer))
                for (int expert = 0; expert < n_expert_; ++expert)
                    candidates.push_back({layer, expert});
    }

    size_t gpu_experts = 0;
    std::vector<size_t> gpu_by_layer(n_layer_, 0);
    std::vector<size_t> ram_by_layer(n_layer_, 0);
    for (const Candidate candidate : candidates) {
        const int layer = candidate.layer;
        const int expert = candidate.expert;
        const int key = layer * n_expert_ + expert;
        bool fits = true;
        for (int role = 0; role < N_ROLE; ++role) {
            const Pool & pool = pools_[layer_pool_[role][layer]];
            if (!pool.key2slot.count(key) &&
                pool.key2slot.size() >= static_cast<size_t>(pool.n_slots)) fits = false;
        }
        if (!fits) continue;
        for (int role = 0; role < N_ROLE; ++role) {
            Pool & pool = pools_[layer_pool_[role][layer]];
            if (!pool.key2slot.count(key))
                install(pool, key, layer, expert, static_cast<Role>(role));
        }
        ++gpu_experts;
        ++gpu_by_layer[layer];
    }
    ggml_backend_synchronize(backend_);

    // Avoid duplicates: the RAM tier holds complete experts absent from GPU.
    // The cache's configured byte cap bounds allocation.
    ExpertIOFiles files;
    const uint64_t capacity = host_cache_->stats().capacity_bytes;
    uint64_t occupied = host_cache_->stats().resident_bytes;
    size_t ram_experts = 0;
    for (const Candidate candidate : candidates) {
        const int layer = candidate.layer;
        const int expert = candidate.expert;
        bool on_gpu = false;
        size_t expert_bytes = 0;
        for (int role = 0; role < N_ROLE; ++role) {
            on_gpu |= resident(static_cast<Role>(role), layer, expert);
            expert_bytes += tensor(static_cast<Role>(role), layer)->nb[2];
        }
        if (on_gpu || expert_bytes > capacity - occupied) continue;
        for (int role = 0; role < N_ROLE; ++role) {
            const Role selected = static_cast<Role>(role);
            const size_t bytes = tensor(selected, layer)->nb[2];
            const std::string & path = fpath_[role][layer];
            const uint64_t offset = foff_[role][layer] + static_cast<uint64_t>(expert) * bytes;
            const bool added = host_cache_->put(ram_key(role, layer, expert), bytes,
                [&](void * destination) {
                    FILE * & file = files.buffered[path];
                    if (!file) file = fopen(path.c_str(), "rb");
                    if (!file) throw std::runtime_error("Cold cache priming could not open GGUF shard");
#ifdef _WIN32
                    if (_fseeki64(file, static_cast<long long>(offset), SEEK_SET) != 0 ||
#else
                    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0 ||
#endif
                        fread(destination, 1, bytes, file) != bytes)
                        throw std::runtime_error("Cold cache priming could not read expert slab");
                });
            if (!added) throw std::runtime_error("Cold cache priming rejected a RAM expert slab");
            occupied += bytes;
        }
        ++ram_experts;
        ++ram_by_layer[layer];
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    fprintf(stderr, "cold expert prime (%s): %zu GPU experts, %zu RAM experts, %.1f MiB RAM, %.3f seconds\n",
            layout.c_str(), gpu_experts, ram_experts, occupied / 1048576.0, seconds);
    for (int layer = 0; layer < n_layer_; ++layer) {
        if (!serves(layer)) continue;
        fprintf(stderr, "cold expert prime layer %d: GPU %zu, RAM %zu, SSD %zu\n",
                layer, gpu_by_layer[layer], ram_by_layer[layer],
                static_cast<size_t>(n_expert_) - gpu_by_layer[layer] - ram_by_layer[layer]);
    }
}

} // namespace questwend
