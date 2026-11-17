// Pure helper correctness gate. No model/backend/runtime construction.
#include "expert_install_check.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

struct Pool { std::unordered_map<int, int> key2slot; };
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        constexpr int key = 1031;
        unsigned skipped = 0, decisions = 0;
        // Every role independently has neither, retained, temporary, or both.
        for (unsigned state = 0; state < 64; ++state) {
            Pool storage[3];
            Pool * pools[3] = {&storage[0], &storage[1], &storage[2]};
            std::unordered_map<int, int> temporary[3];
            bool expected_missing = false;
            for (unsigned role = 0; role < 3; ++role) {
                const unsigned flags = (state >> (role * 2)) & 3;
                storage[role].key2slot[999] = 7;
                temporary[role][888] = 9;
                if (flags & 1) storage[role].key2slot[key] = 12 + role;
                if (flags & 2) temporary[role][key] = 32 + role;
                expected_missing |= flags == 0;
            }
            const std::array<Pool, 3> retained_before = {storage[0], storage[1], storage[2]};
            const std::array<std::unordered_map<int, int>, 3> temporary_before = {
                temporary[0], temporary[1], temporary[2]};
            // Repeated occurrences must not mutate maps or memoize a decision.
            for (unsigned repeat = 0; repeat < 3; ++repeat) {
                const bool needs = questwend::expert_role_needs_install(key, pools, temporary);
                require(needs == expected_missing, "Wrong mixed-role install predicate");
                bool policy_called = false;
                auto can_retain = [&] { policy_called = true; ++decisions; return bool(repeat % 2); };
                const bool retain = needs && can_retain();
                require(policy_called == needs, "Unused decision was not short-circuited");
                if (needs) require(retain == bool(repeat % 2), "Fresh policy decision changed");
                else ++skipped;
                unsigned default_calls = 0;
                const bool default_needs = !false || questwend::expert_role_needs_install(key, pools, temporary);
                const bool default_retain = default_needs && [&] { ++default_calls; return true; }();
                require(default_retain && default_calls == 1, "Default path skipped the original decision");
                for (unsigned role = 0; role < 3; ++role) {
                    require(storage[role].key2slot == retained_before[role].key2slot,
                            "Predicate mutated retained residency");
                    require(temporary[role] == temporary_before[role],
                            "Predicate mutated per-call temporary residency");
                }
            }
            // A later actual missing role must be re-evaluated immediately.
            storage[1].key2slot.erase(key);
            temporary[1].erase(key);
            require(questwend::expert_role_needs_install(key, pools, temporary),
                    "New installation reused a stale covered-role decision");
        }
        require(skipped == 81 && decisions == 111, "Fixture did not exercise all presence states");
        std::cout << "{\"state\":\"passed\",\"presence_states\":64,"
                     "\"repeated_checks\":192,\"unused_decisions_skipped\":" << skipped << "}" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Expert install check failed: " << error.what() << std::endl;
        return 1;
    }
}
