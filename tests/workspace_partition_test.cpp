#include "../core/workspace_partition.h"
#include "../core/expert_admission.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

using namespace questwend;

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

template<class Function>
static void rejects(Function function, const char * message) {
    bool rejected = false;
    try { function(); } catch (const std::exception &) { rejected = true; }
    check(rejected, message);
}

struct PoolState {
    int n_slots = 8, scratch_slots = 4;
    std::vector<int> slot2key = std::vector<int>(8, -1);
    std::vector<uint64_t> clk = std::vector<uint64_t>(8, 0);
    std::map<int, int> key2slot;
    std::vector<int> slot_of = std::vector<int>(32, 0);
    std::vector<int> occupancy = std::vector<int>(2, 0);
    // A physical pool never reallocates or changes shape during repartition.
    std::vector<int> physical = std::vector<int>(12, -1);
};

static void install(PoolState & pool, int key, int slot) {
    check(pool.slot2key.at(slot) == -1, "Mock install overwrote a live slot");
    pool.slot2key[slot] = key;
    pool.key2slot[key] = slot;
    pool.slot_of.at(key) = slot;
    pool.clk[slot] = key + 100;
    ++pool.occupancy.at(key / 16);
    pool.physical.at(slot) = key;
}

static void validate(const std::vector<PoolState> & pools) {
    for (const auto & pool : pools) {
        check(pool.n_slots + pool.scratch_slots == 12, "Physical size changed");
        check(pool.slot2key.size() == (size_t) pool.n_slots, "Slot vector length mismatch");
        std::vector<int> counts(2, 0);
        for (int slot = 0; slot < pool.n_slots; ++slot) {
            const int key = pool.slot2key[slot];
            if (key < 0) continue;
            check(pool.key2slot.at(key) == slot && pool.slot_of.at(key) == slot,
                  "Reverse map mismatch");
            check(pool.physical.at(slot) == key, "Weight identity changed");
            ++counts.at(key / 16);
        }
        check(counts == pool.occupancy, "Layer occupancy mismatch");
        for (int key = 0; key < 32; ++key)
            if (!pool.key2slot.count(key)) check(pool.slot_of[key] == 0, "Stale sentinel");
    }
}

int main() {
    std::vector<PoolState> pools(3);
    for (auto & pool : pools) install(pool, 6, 1);
    install(pools[0], 21, 7);
    install(pools[1], 21, 2); // Peer below the cut must also be removed.
    install(pools[2], 21, 6);
    validate(pools);
    const auto physical_before = std::vector<std::vector<int>>{
        pools[0].physical, pools[1].physical, pools[2].physical};
    std::map<std::pair<size_t, int>, int> demoted;
    auto release_whole_expert = [&](size_t index, int slot) {
        const int key = pools.at(index).slot2key.at(slot);
        for (size_t role = 0; role < pools.size(); ++role) {
            auto & pool = pools[role];
            const auto found = pool.key2slot.find(key);
            if (found == pool.key2slot.end()) continue;
            const int peer_slot = found->second;
            demoted[{role, key}] = pool.physical.at(peer_slot);
            pool.slot2key[peer_slot] = -1;
            pool.slot_of[key] = 0;
            --pool.occupancy.at(key / 16);
            pool.key2slot.erase(found);
        }
    };
    auto plan = plan_workspace_partition(pools, 8, 2);
    check(plan.changed && plan.retained[0] == 4, "Wrong prefill partition");
    require_workspace_boundary(std::vector<uint8_t>(32, 0));
    retire_workspace_tail(pools, plan, release_whole_expert);
    commit_workspace_partition(pools, plan);
    validate(pools);
    check(demoted.size() == 3 && pools[1].slot2key[2] == -1, "Whole-expert eviction failed");
    for (const auto & item : demoted) check(item.second == 21, "Demoted wrong weights");
    for (const auto & pool : pools) check(pool.key2slot.at(6) == 1, "Kept mapping moved");

    plan = plan_workspace_partition(pools, 2, 2);
    retire_workspace_tail(pools, plan, release_whole_expert);
    commit_workspace_partition(pools, plan);
    validate(pools);
    for (size_t role = 0; role < pools.size(); ++role) {
        check(pools[role].physical == physical_before[role], "Physical buffer changed");
        for (int slot = 4; slot < 10; ++slot)
            check(pools[role].slot2key[slot] == -1 && pools[role].clk[slot] == 0,
                  "New retained slots are not empty");
    }
    check(!plan_workspace_partition(pools, 2, 2).changed, "No-op transition changed state");
    rejects([&] { plan_workspace_partition(pools, 11, 2); }, "Oversized scratch accepted");
    std::vector<uint8_t> protected_ids(32, 0);
    protected_ids[6] = 1;
    rejects([&] { require_workspace_boundary(protected_ids); }, "Active selection accepted");
    install(pools[0], 7, 9);
    const auto unsafe = plan_workspace_partition(pools, 8, 2);
    rejects([&] { commit_workspace_partition(pools, unsafe); }, "Live tail truncated");
    check(pools[0].key2slot.at(7) == 9 && pools[0].n_slots == 10,
          "Rejected commit modified metadata");

    ExpertAdmission admission(32, 1024);
    for (int index = 0; index < 3000; ++index) admission.observe(6);
    admission.set_capacity(2048);
    check(admission.frequency(6) == 3000, "Capacity growth erased history");
    admission.set_capacity(256);
    check(admission.frequency(6) == 1500, "Capacity shrink did not age to horizon");
    std::cout << "workspace partition metadata gates passed\n";
}
