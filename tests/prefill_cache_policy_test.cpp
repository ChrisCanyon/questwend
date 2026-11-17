#include "../core/prefill_cache_policy.h"
#include "../core/expert_admission.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace questwend;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static std::vector<uint32_t> frequencies(const ExpertAdmission & admission) {
    std::vector<uint32_t> result;
    for (size_t expert = 0; expert < 32; ++expert) result.push_back(admission.frequency(expert));
    return result;
}

int main() {
    PrefillCachePolicy disabled;
    for (bool batched : {false, true, false}) {
        disabled.enter_phase(batched);
        check(disabled.record_decode_frequency() && disabled.admit_new_ssd_copy(),
              "Flag-off behavior changed");
    }

    PrefillCachePolicy policy;
    policy.set_enabled(true);
    ExpertAdmission admission(32, 1024);
    for (int count = 0; count < 6000; ++count) admission.observe(6);
    for (int count = 0; count < 1500; ++count) admission.observe(7);
    const auto decode_history = frequencies(admission);
    check(policy.record_decode_frequency() && policy.admit_new_ssd_copy(),
          "Initial decode phase is gated");

    policy.enter_phase(true);
    // Shrinking retained capacity would ordinarily age samples immediately.
    // The real set_phase_workspace uses this same gate before set_capacity.
    if (policy.record_decode_frequency()) admission.set_capacity(256);
    uint64_t actual_loads = 0, actual_computations = 0, activation_observations = 0;
    uint64_t new_ram_admissions = 0;
    // Model the source ordering: activation observations and selected expert
    // execution are independent of the two cache-only decisions.
    for (int sample = 0; sample < 2000000; ++sample) {
        const size_t expert = sample % 32;
        ++activation_observations;
        if (policy.record_decode_frequency()) admission.observe(expert);
        ++actual_loads;
        if (policy.admit_new_ssd_copy()) ++new_ram_admissions;
        ++actual_computations;
    }
    check(frequencies(admission) == decode_history, "Prefill aged decode history");
    check(actual_loads == 2000000 && actual_computations == 2000000,
          "Cache gate skipped selected computation");
    check(activation_observations == 2000000, "Activation trace was gated");
    check(new_ram_admissions == 0, "Cold prefill copies were admitted");

    policy.enter_phase(false);
    if (policy.record_decode_frequency()) admission.set_capacity(2048);
    check(policy.record_decode_frequency() && policy.admit_new_ssd_copy(),
          "Decode did not restore normal caching");
    if (policy.record_decode_frequency()) admission.observe(6);
    check(admission.frequency(6) == decode_history[6] + 1, "Decode history did not resume");
    const auto next_decode_history = frequencies(admission);
    policy.enter_phase(true);
    for (int count = 0; count < 4000; ++count)
        if (policy.record_decode_frequency()) admission.observe(9);
    check(frequencies(admission) == next_decode_history, "Second prefill changed history");

    policy.set_enabled(false);
    check(policy.record_decode_frequency() && policy.admit_new_ssd_copy(),
          "Disabling policy did not restore baseline");
    std::cout << "prefill cache policy gates passed\n";
}
