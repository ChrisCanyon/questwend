#pragma once

namespace questwend {

// Cache policy only: neither flag is permission to skip an expert computation.
// Genuine source prefill has different locality from generation. A batched
// speculative verify is still generation, not source prefill.
class PrefillCachePolicy {
public:
    void set_enabled(bool enabled) { enabled_ = enabled; }
    void enter_phase(bool source_prefill) { source_prefill_ = source_prefill; }
    bool record_decode_frequency() const { return !enabled_ || !source_prefill_; }
    bool admit_new_ssd_copy() const { return !enabled_ || !source_prefill_; }

private:
    bool enabled_ = false;
    bool source_prefill_ = false;
};

} // namespace questwend
