#pragma once

// Optional RAM-copy admission only. Expert computation and GPU placement are
// unchanged. One logical expert frequency supplies the same three-role decision.
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace questwend {

class RamSsdAdmission {
public:
    explicit RamSsdAdmission(uint32_t minimum_hits = 0, bool has_frequency_counter = true)
        : minimum_hits_(minimum_hits) {
        if (enabled() && !has_frequency_counter)
            throw std::invalid_argument("QWEN_RAM_SSD_MIN_HITS requires logical expert admission counters");
    }

    static uint32_t parse_threshold(const char * value) {
        if (!value || !*value) return 0;
        uint32_t result = 0;
        for (const char * digit = value; *digit; ++digit) {
            if (*digit < '0' || *digit > '9')
                throw std::invalid_argument("QWEN_RAM_SSD_MIN_HITS must be an unsigned decimal integer");
            const uint32_t next = static_cast<uint32_t>(*digit - '0');
            if (result > (std::numeric_limits<uint32_t>::max() - next) / 10)
                throw std::invalid_argument("QWEN_RAM_SSD_MIN_HITS exceeds uint32 range");
            result = result * 10 + next;
        }
        return result;
    }

    static RamSsdAdmission from_environment(bool has_frequency_counter) {
        return RamSsdAdmission(parse_threshold(std::getenv("QWEN_RAM_SSD_MIN_HITS")),
                               has_frequency_counter);
    }

    bool enabled() const { return minimum_hits_ != 0; }
    uint32_t minimum_hits() const { return minimum_hits_; }
    bool accepts(uint32_t logical_hits) const { return !enabled() || logical_hits >= minimum_hits_; }

private:
    uint32_t minimum_hits_ = 0;
};

} // namespace questwend
