#include "../core/workspace_partition.h"
#include "../core/prefill_cache_policy.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

using namespace questwend;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
static void rejects(Function function, const char * message) {
    bool rejected = false;
    try { function(); } catch (const std::exception &) { rejected = true; }
    check(rejected, message);
}

struct Pool {
    int n_slots = 1014, scratch_slots = 10;
    std::vector<int> slot2key = std::vector<int>(1014, -1);
    std::vector<uint64_t> clk = std::vector<uint64_t>(1014, 0);
    std::map<int, int> key2slot;
    std::vector<int> slot_of = std::vector<int>(512, 0);
};

static void install(Pool & pool, int expert, int slot) {
    pool.slot2key.at(slot) = expert;
    pool.clk.at(slot) = 9;
    pool.key2slot[expert] = slot;
    pool.slot_of.at(expert) = slot;
}

static void validate(const std::vector<Pool> & pools) {
    for (const auto & pool : pools) {
        check(pool.n_slots + pool.scratch_slots == 1024, "Physical tensor capacity changed");
        check(pool.slot2key.size() == static_cast<size_t>(pool.n_slots), "Retained metadata length changed incorrectly");
        for (const auto & entry : pool.key2slot) {
            check(pool.slot2key.at(entry.second) == entry.first, "Reverse map became invalid");
            check(pool.slot_of.at(entry.first) == entry.second, "Slot sentinel became invalid");
        }
    }
}

int main() {
    // Kernel batching determines the scratch union; genuine source prefill
    // independently determines whether decode history/RAM admission are frozen.
    PrefillCachePolicy default_policy;
    for (bool source_prefill : {false, true, false}) {
        default_policy.enter_phase(source_prefill);
        check(default_policy.record_decode_frequency() && default_policy.admit_new_ssd_copy(),
              "Default flag-off policy changed");
    }
    PrefillCachePolicy preserved;
    preserved.set_enabled(true);
    preserved.enter_phase(true);
    check(!preserved.record_decode_frequency() && !preserved.admit_new_ssd_copy(),
          "Genuine source prefill did not freeze history/admission");
    for (int verify_positions : {2, 4, 9}) {
        preserved.enter_phase(false);  // decode_cached_batch(..., verify=true)
        check(bounded_workspace_scratch(512, 10, verify_positions) == 10 * verify_positions,
              "Verification batch did not reserve the bounded expert union");
        check(preserved.record_decode_frequency() && preserved.admit_new_ssd_copy(),
              "Verification batch was incorrectly treated as source prefill");
    }
    preserved.enter_phase(true);
    check(bounded_workspace_scratch(512, 10, 4096) == 512 &&
          !preserved.record_decode_frequency() && !preserved.admit_new_ssd_copy(),
          "Large source prefill scratch/cache policy is incorrect");
    preserved.enter_phase(false);  // ordinary scalar generation
    check(bounded_workspace_scratch(512, 10, 1) == 10 &&
          preserved.record_decode_frequency() && preserved.admit_new_ssd_copy(),
          "Scalar decode did not resume generation cache behavior");
    check(bounded_workspace_scratch(512, 10, 1) == 10, "Decode selection bound incorrect");
    for (int tokens = 2; tokens <= 9; ++tokens)
        check(bounded_workspace_scratch(512, 10, tokens) == 10 * tokens, "Tiny verify bound incorrect");
    check(bounded_workspace_scratch(512, 10, 4096) == 512, "Large prefill cap incorrect");
    check(bounded_workspace_scratch(511, 10, 51) == 510, "Nondivisible near-cap incorrect");
    check(bounded_workspace_scratch(511, 10, 52) == 511, "Nondivisible saturation incorrect");
    const int maximum = std::numeric_limits<int>::max();
    check(bounded_workspace_scratch(maximum, maximum, maximum) == maximum, "Overflow guard failed");
    check(bounded_workspace_scratch(maximum, maximum / 2, 3) == maximum, "Large product overflow guard failed");
    rejects([] { bounded_workspace_scratch(512, 10, 0); }, "Zero token count accepted");
    rejects([] { bounded_workspace_scratch(512, 513, 1); }, "Top-k exceeding expert count accepted");

    std::vector<Pool> pools(3);
    for (auto & pool : pools) install(pool, 6, 1);
    // A role lies in the newly borrowed scratch tail; every peer role must leave.
    install(pools[0], 21, 1010);
    install(pools[1], 21, 2);
    install(pools[2], 21, 1008);
    auto release_whole_expert = [&](size_t role, int slot) {
        const int expert = pools.at(role).slot2key.at(slot);
        for (auto & pool : pools) {
            const auto found = pool.key2slot.find(expert);
            if (found == pool.key2slot.end()) continue;
            pool.slot2key.at(found->second) = -1;
            pool.clk.at(found->second) = 0;
            pool.slot_of.at(expert) = 0;
            pool.key2slot.erase(found);
        }
    };
    for (int tokens : {2, 4, 9, 4096, 1, 8, 1}) {
        const int scratch = bounded_workspace_scratch(512, 10, tokens);
        auto plan = plan_workspace_partition(pools, scratch, 10);
        require_workspace_boundary(std::vector<uint8_t>(512, 0));
        retire_workspace_tail(pools, plan, release_whole_expert);
        commit_workspace_partition(pools, plan);
        validate(pools);
        for (const auto & pool : pools) {
            check(pool.scratch_slots == scratch, "Wrong scratch slots after transition");
            check(pool.key2slot.at(6) == 1, "Unrelated retained expert moved");
            check(!pool.key2slot.count(21) && pool.slot_of[21] == 0, "Evicted peer role/sentinel survived");
        }
    }
    std::vector<uint8_t> active(512, 0);
    active[6] = 1;
    rejects([&] { require_workspace_boundary(active); }, "Protected selection allowed repartition");
    install(pools[0], 7, 1010);
    const auto unsafe = plan_workspace_partition(pools, 20, 10);
    rejects([&] { commit_workspace_partition(pools, unsafe); }, "Live expert truncated without demotion");
    check(pools[0].key2slot.at(7) == 1010, "Rejected commit changed retained expert");
    std::cout << "tiny-batch workspace metadata gates passed\n";
}
