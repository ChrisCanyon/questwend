#include "ram_ssd_admission.h"
#include "expert_admission.h"

#include <iostream>
#include <stdexcept>

using questwend::ExpertAdmission;
using questwend::RamSsdAdmission;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
static void rejects_invalid(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "expected invalid argument rejection");
}

static void parser_and_disabled_defaults() {
    check(RamSsdAdmission::parse_threshold(nullptr) == 0, "unset default");
    check(RamSsdAdmission::parse_threshold("") == 0, "empty default");
    check(RamSsdAdmission::parse_threshold("0") == 0, "zero default");
    check(RamSsdAdmission::parse_threshold("0002") == 2, "leading zeros");
    check(RamSsdAdmission::parse_threshold("4294967295") == UINT32_MAX, "maximum unsigned value");
    for (const char * invalid : {"-1", "+2", " 2", "2 ", "1.0", "2x", "4294967296"})
        rejects_invalid([&] { RamSsdAdmission::parse_threshold(invalid); });
    RamSsdAdmission disabled(0, false);
    check(!disabled.enabled() && disabled.accepts(0) && disabled.accepts(UINT32_MAX), "flag-off path");
    rejects_invalid([] { RamSsdAdmission unsupported(2, false); });
}

static void all_roles_share_one_logical_decision() {
    const size_t expert = 12 * 512 + 9;
    for (uint32_t minimum : {2u, 3u, 4u}) {
        ExpertAdmission frequencies(48 * 512, 256);
        RamSsdAdmission policy(minimum);
        for (uint32_t observations = 0; observations < 5; ++observations) {
            const bool expected = frequencies.frequency(expert) >= minimum;
            for (int role = 0; role < 3; ++role)
                check(policy.accepts(frequencies.frequency(expert)) == expected, "inconsistent three-role admission");
            frequencies.observe(expert);
        }
    }
}

static void frequency_aging_changes_admission_without_changing_weights() {
    ExpertAdmission frequencies(2, 256);
    RamSsdAdmission policy(2);
    frequencies.observe(0);
    check(!policy.accepts(frequencies.frequency(0)), "first touch should not qualify");
    frequencies.observe(0);
    check(policy.accepts(frequencies.frequency(0)), "second touch should qualify");
    for (int count = 0; count < 2558; ++count) frequencies.observe(1);
    check(frequencies.frequency(0) == 1, "expected existing TinyLFU aging");
    check(!policy.accepts(frequencies.frequency(0)), "aged expert should use new frequency");
    frequencies.observe(0);
    check(policy.accepts(frequencies.frequency(0)), "expert should qualify again after demand");
}

int main() {
    try {
        parser_and_disabled_defaults();
        all_roles_share_one_logical_decision();
        frequency_aging_changes_admission_without_changing_weights();
        std::cout << "RAM SSD admission tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "RAM SSD admission test failed: " << error.what() << '\n';
        return 1;
    }
}
