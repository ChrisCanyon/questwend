#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace questwend {

// At most tokens * top-k different experts can be selected by one layer.
// Cap before multiplying, so large valid prefill lengths cannot overflow int.
inline int bounded_workspace_scratch(int experts, int used, int tokens) {
    if (experts < 1 || used < 1 || used > experts || tokens < 1)
        throw std::invalid_argument("Invalid expert workspace selection dimensions");
    if (tokens > experts / used) return experts;
    return used * tokens;  // Proven <= experts <= INT_MAX above.
}

// Pure metadata planning shared with the native state tests. The physical
// tensor never changes: retained slots are a prefix and scratch is its tail.
struct WorkspacePartition {
    std::vector<int> retained;
    int scratch = 0;
    bool changed = false;
};

template<class Flags>
void require_workspace_boundary(const Flags & active) {
    for (auto flag : active)
        if (flag) throw std::logic_error("Cannot repartition an active expert selection");
}

template<class Pools>
WorkspacePartition plan_workspace_partition(const Pools & pools, int scratch, int minimum_retained) {
    if (scratch < 1 || minimum_retained < 1)
        throw std::invalid_argument("Invalid expert workspace dimensions");
    WorkspacePartition plan;
    plan.scratch = scratch;
    for (const auto & pool : pools) {
        if (pool.n_slots < 0 || pool.scratch_slots < 0 ||
            pool.slot2key.size() != (size_t) pool.n_slots ||
            pool.clk.size() != (size_t) pool.n_slots)
            throw std::logic_error("Invalid retained expert slot metadata");
        const int retained = pool.n_slots + pool.scratch_slots - scratch;
        if (retained < minimum_retained)
            throw std::runtime_error("GPU pool cannot fit expert workspace plus retained selection");
        plan.retained.push_back(retained);
        plan.changed = plan.changed || pool.scratch_slots != scratch;
    }
    return plan;
}

// Call only after all graph accesses have completed. The callback must remove
// the complete logical expert (all role pools), including reverse maps, LRU,
// occupancy and sentinels. Do not resize any pool until every callback finishes.
template<class Pools, class Release>
void retire_workspace_tail(Pools & pools, const WorkspacePartition & plan, Release release) {
    if (pools.size() != plan.retained.size())
        throw std::logic_error("Expert workspace pool count changed");
    for (size_t index = 0; index < pools.size(); ++index) {
        auto & pool = pools[index];
        for (int slot = plan.retained[index]; slot < pool.n_slots; ++slot)
            if (pool.slot2key[slot] >= 0) release(index, slot);
    }
}

template<class Pools>
void commit_workspace_partition(Pools & pools, const WorkspacePartition & plan) {
    // Validate all tails before changing any index base.
    for (size_t index = 0; index < pools.size(); ++index) {
        const auto & pool = pools[index];
        for (int slot = plan.retained[index]; slot < pool.n_slots; ++slot)
            if (pool.slot2key[slot] >= 0)
                throw std::logic_error("Live expert remains in new transient workspace");
    }
    for (size_t index = 0; index < pools.size(); ++index) {
        auto & pool = pools[index];
        pool.slot2key.resize(plan.retained[index], -1);
        pool.clk.resize(plan.retained[index], 0);
        pool.n_slots = plan.retained[index];
        pool.scratch_slots = plan.scratch;
    }
}

} // namespace questwend
