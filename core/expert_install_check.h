#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace questwend {

inline bool skip_unused_retain_requested() {
    static const bool enabled = [] {
        const char * value = std::getenv("QWEN_SKIP_UNUSED_RETAIN");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// Owner-only observation at the original decision point. This does not reserve
// slots, touch clocks, remember decisions or change the per-occurrence updates.
template<typename Pool, typename Temporary, std::size_t Roles>
bool expert_role_needs_install(int key, Pool * (&pools)[Roles],
                               const Temporary (&temporary)[Roles]) {
    for (std::size_t role = 0; role < Roles; ++role) {
        if (pools[role]->key2slot.find(key) == pools[role]->key2slot.end() &&
            temporary[role].find(key) == temporary[role].end())
            return true;
    }
    return false;
}

} // namespace questwend
