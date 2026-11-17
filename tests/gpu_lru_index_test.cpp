#include "gpu_lru_index.h"
#include "gpu_lru_pool_hooks.h"
#include "workspace_partition.h"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <memory>
#include <unordered_map>

static std::atomic<bool> forbid_allocation{false};
void * operator new(std::size_t bytes) {
    if (forbid_allocation.load()) throw std::bad_alloc();
    if (void * allocation = std::malloc(bytes ? bytes : 1)) return allocation;
    throw std::bad_alloc();
}
void * operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void * allocation) noexcept { std::free(allocation); }
void operator delete[](void * allocation) noexcept { std::free(allocation); }
void operator delete(void * allocation, std::size_t) noexcept { std::free(allocation); }
void operator delete[](void * allocation, std::size_t) noexcept { std::free(allocation); }

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

// Exact unconstrained/activation-disabled reference, including lowest-index
// free precedence and the UINT64_MAX strict-comparison edge in existing code.
static int linear_scan(int count, const std::vector<int> & keys,
    const std::vector<uint64_t> & clocks, const std::vector<uint8_t> & protected_slots) {
    int victim = -1;
    uint64_t best = std::numeric_limits<uint64_t>::max();
    for (int slot = 0; slot < count; ++slot) {
        if (keys[slot] < 0) return slot;
        if (protected_slots[slot]) continue;
        if (clocks[slot] < best) { best = clocks[slot]; victim = slot; }
    }
    return victim;
}

static void compare(questwend::GpuLruIndex & index, int count,
    const std::vector<int> & keys, const std::vector<uint64_t> & clocks,
    const std::vector<uint8_t> & protected_slots) {
    const auto before = index.pick([&](int slot) { return protected_slots[slot] != 0; });
    const auto again = index.pick([&](int slot) { return protected_slots[slot] != 0; });
    check(before.slot == linear_scan(count, keys, clocks, protected_slots), "Indexed victim differs from original scan");
    check(before.slot == again.slot && before.occupied_examined == again.occupied_examined,
        "Selection mutated LRU/protected ordering");
    check(index.invariant(keys, clocks), "GPU LRU index metadata invariant failed");
}

static void boundaries_ties_and_protection() {
    constexpr int count = 130;
    questwend::GpuLruIndex index(count);
    std::vector<int> keys(count);
    std::vector<uint64_t> clocks(count, 11);
    std::vector<uint8_t> protected_slots(count, 0);
    for (int slot = 0; slot < count; ++slot) keys[slot] = slot;
    index.rebuild(count, keys, clocks);
    compare(index, count, keys, clocks, protected_slots);
    check(index.pick([](int) { return false; }).slot == 0, "Clock tie did not select the lowest slot");
    protected_slots[0] = protected_slots[1] = 1;
    check(index.pick([&](int slot) { return protected_slots[slot] != 0; }).slot == 2,
        "Selected active-demand slot");
    for (int slot : {129, 64, 63}) { keys[slot] = -1; index.release(slot); }
    check(index.pick([](int) { return true; }).slot == 63, "Lowest free bitmap slot precedence changed");
    for (int slot : {63, 64, 129}) {
        keys[slot] = slot;
        clocks[slot] = 2;
        index.occupy_or_touch(slot, clocks[slot]);
        compare(index, count, keys, clocks, protected_slots);
    }
    std::fill(protected_slots.begin(), protected_slots.end(), 1);
    check(index.pick([](int) { return true; }).slot == -1, "All-protected pool produced a victim");
    std::fill(protected_slots.begin(), protected_slots.end(), 0);
    std::fill(clocks.begin(), clocks.end(), std::numeric_limits<uint64_t>::max());
    index.rebuild(count, keys, clocks);
    compare(index, count, keys, clocks, protected_slots);
    check(index.pick([](int) { return false; }).slot == -1, "UINT64_MAX scan edge changed");
    clocks[80] = 0; // Wrapped global clock must order before the max-clock slots.
    index.occupy_or_touch(80, 0);
    compare(index, count, keys, clocks, protected_slots);
}

static void randomized_rebuild_and_no_allocations() {
    constexpr int maximum = 257;
    questwend::GpuLruIndex index(maximum);
    std::vector<int> keys(maximum, -1);
    std::vector<uint64_t> clocks(maximum, 0);
    std::vector<uint8_t> protected_slots(maximum, 0);
    int count = maximum;
    index.rebuild(count, keys, clocks);
    uint64_t random = 39, clock = 1;
    auto next = [&]() {
        random = random * 6364136223846793005ull + 1442695040888963407ull;
        return random >> 32;
    };
    forbid_allocation = true;
    for (int operation = 0; operation < 10000; ++operation) {
        const unsigned action = static_cast<unsigned>(next() % 8);
        const int slot = static_cast<int>(next() % count);
        if (action == 0) { keys[slot] = -1; index.release(slot); }
        else if (action == 1) {
            const int new_count = 1 + static_cast<int>(next() % maximum);
            if (new_count > count)
                for (int added = count; added < new_count; ++added) { keys[added] = -1; clocks[added] = 0; }
            count = new_count;
            index.rebuild(count, keys, clocks);
        } else {
            keys[slot] = slot + 1000;
            clocks[slot] = action == 2 ? next() % 4 : ++clock; // Include repeated/out-of-order clocks.
            index.occupy_or_touch(slot, clocks[slot]);
        }
        protected_slots[next() % count] ^= 1;
        compare(index, count, keys, clocks, protected_slots);
    }
    forbid_allocation = false;
}

struct HookPool {
    int n_slots, scratch_slots;
    std::vector<int> slot2key, slot_of, occ;
    std::vector<uint64_t> clk;
    std::unordered_map<int, int> key2slot;
    std::unique_ptr<questwend::GpuLruIndex> lru_index;
    HookPool(int physical, int retained, bool indexed)
        : n_slots(retained), scratch_slots(physical - retained),
          slot2key(retained, -1), slot_of(64, 0), occ(4, 0), clk(retained, 0) {
        slot2key.reserve(physical);
        clk.reserve(physical);
        if (indexed) {
            lru_index = std::make_unique<questwend::GpuLruIndex>(physical);
            lru_index->rebuild(n_slots, slot2key, clk);
        }
    }
};

static void check_hook_maps(const HookPool & pool) {
    size_t count = 0;
    int layer_counts[4] = {};
    for (int slot = 0; slot < pool.n_slots; ++slot) {
        const int key = pool.slot2key[slot];
        if (key < 0) continue;
        const auto found = pool.key2slot.find(key);
        check(found != pool.key2slot.end() && found->second == slot && pool.slot_of[key] == slot,
            "Shared mutation hook left inconsistent authoritative maps");
        ++layer_counts[key / 16];
        ++count;
    }
    check(count == pool.key2slot.size(), "Shared mutation hook left an orphan reverse key");
    for (size_t layer = 0; layer < pool.occ.size(); ++layer)
        check(pool.occ[layer] == layer_counts[layer], "Shared mutation hook changed layer occupancy");
    if (pool.lru_index)
        check(pool.lru_index->invariant(pool.slot2key, pool.clk), "Shared mutation hook left stale index metadata");
}

static void actual_publication_touch_and_drop_hooks(bool indexed) {
    HookPool pool(8, 6, indexed);
    uint64_t clock = 0;
    questwend::publish_gpu_lru_slot(pool, 0, 7, 0, clock);
    questwend::publish_gpu_lru_slot(pool, 1, 18, 1, clock);
    check(clock == 2 && pool.clk[0] == 1 && pool.clk[1] == 2, "Publication clock sequence changed");
    questwend::record_gpu_lru_access(pool, 0, clock);
    check(clock == 3 && pool.clk[0] == 3, "Resident-hit shared clock hook failed");
    check_hook_maps(pool);
    check(questwend::drop_gpu_lru_slot(pool, 0, 16) == 7, "Shared removal returned a wrong key");
    check(pool.slot_of[7] == 0 && !pool.key2slot.count(7), "Shared removal kept stale remap ownership");
    check_hook_maps(pool);

    // New map publication is the only allocating mutation. Force it to fail:
    // no slot, occupancy, global clock, remap or index entry may be published.
    bool threw = false;
    forbid_allocation = true;
    try { questwend::publish_gpu_lru_slot(pool, 2, 33, 2, clock); }
    catch (const std::bad_alloc &) { threw = true; }
    forbid_allocation = false;
    check(threw, "Map-publication fault was not exercised");
    check(clock == 3 && pool.slot2key[2] == -1 && pool.slot_of[33] == 0 && !pool.key2slot.count(33),
        "Failed map allocation changed authoritative slot state");
    check_hook_maps(pool);
    questwend::publish_gpu_lru_slot(pool, 2, 33, 2, clock);
    check_hook_maps(pool);
}

static void whole_expert_phase_mutation_hooks() {
    std::vector<HookPool> pools;
    pools.reserve(3);
    for (int role = 0; role < 3; ++role) pools.emplace_back(8, 6, true);
    uint64_t clock = 0;
    // One complete expert occupies different slots across its role pools.
    questwend::publish_gpu_lru_slot(pools[0], 0, 7, 0, clock);
    questwend::publish_gpu_lru_slot(pools[1], 1, 7, 0, clock);
    questwend::publish_gpu_lru_slot(pools[2], 5, 7, 0, clock);
    auto plan = questwend::plan_workspace_partition(pools, 4, 1); // retained prefix shrinks6->4.
    questwend::retire_workspace_tail(pools, plan, [&](size_t index, int slot) {
        const int key = pools[index].slot2key[slot];
        for (auto & other : pools) {
            const auto found = other.key2slot.find(key);
            if (found != other.key2slot.end()) questwend::drop_gpu_lru_slot(other, found->second, 16);
        }
    });
    forbid_allocation = true;
    questwend::commit_workspace_partition(pools, plan);
    for (auto & pool : pools) pool.lru_index->rebuild(pool.n_slots, pool.slot2key, pool.clk);
    forbid_allocation = false;
    for (const auto & pool : pools) {
        check(!pool.key2slot.count(7) && pool.n_slots == 4, "Phase tail kept a partial peer-role expert");
        check_hook_maps(pool);
        check(pool.lru_index->pick([](int) { return false; }).slot == 0, "Shrink changed lowest-free ordering");
    }
    plan = questwend::plan_workspace_partition(pools, 1, 1); // retained prefix grows4->7.
    forbid_allocation = true;
    questwend::commit_workspace_partition(pools, plan);
    for (auto & pool : pools) pool.lru_index->rebuild(pool.n_slots, pool.slot2key, pool.clk);
    forbid_allocation = false;
    for (auto & pool : pools) {
        questwend::publish_gpu_lru_slot(pool, 6, 19, 1, clock);
        check_hook_maps(pool);
    }
}

static void stale_index_fails_before_mutation() {
    HookPool pool(8, 6, true);
    uint64_t clock = 0;
    questwend::publish_gpu_lru_slot(pool, 0, 7, 0, clock);
    pool.lru_index->release(0); // Deliberate mismatch: occupied payload, free index.
    bool access_failed = false, drop_failed = false, publish_failed = false;
    try { questwend::record_gpu_lru_access(pool, 0, clock); }
    catch (const std::logic_error &) { access_failed = true; }
    try { questwend::drop_gpu_lru_slot(pool, 0, 16); }
    catch (const std::logic_error &) { drop_failed = true; }
    check(access_failed && drop_failed && clock == 1 && pool.clk[0] == 1 &&
        pool.slot2key[0] == 7 && pool.key2slot.at(7) == 0 && pool.occ[0] == 1 &&
        !pool.lru_index->is_occupied(0), "Stale free index was silently repaired or payload mutated");
    pool.lru_index->occupy_or_touch(0, 1);
    pool.lru_index->occupy_or_touch(2, 99); // Opposite mismatch: free payload, occupied index.
    try { questwend::publish_gpu_lru_slot(pool, 2, 33, 2, clock); }
    catch (const std::logic_error &) { publish_failed = true; }
    check(publish_failed && clock == 1 && pool.slot2key[2] == -1 &&
        !pool.key2slot.count(33) && pool.occ[2] == 0 && pool.lru_index->is_occupied(2),
        "Stale occupied index was repaired after map/scalar publication");
    pool.lru_index->release(2);
    check_hook_maps(pool);
}

int main() {
    try {
        boundaries_ties_and_protection();
        randomized_rebuild_and_no_allocations();
        actual_publication_touch_and_drop_hooks(false);
        actual_publication_touch_and_drop_hooks(true);
        whole_expert_phase_mutation_hooks();
        stale_index_fails_before_mutation();
        std::cout << "GPU LRU indexed/reference equivalence checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        forbid_allocation = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
