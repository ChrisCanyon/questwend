#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace questwend {

// Owner-thread policy only. Workers receive an immutable bool snapshot; this
// class does not make concurrent calls into one ExpertCache thread-safe.
class ExpertSsdReadPhase {
public:
    struct ScopeCounts {
        uint64_t source_direct = 0, source_buffered = 0;
        uint64_t non_source_direct = 0, non_source_buffered = 0;
    };
    explicit ExpertSsdReadPhase(const char * value = std::getenv("QWEN_SSD_PREFILL_BUFFERED"))
        : buffered_source_(parse(value)) {}

    void enter_phase(bool source_prefill) {
        if (active_fetches_ != 0)
            throw std::logic_error("SSD read phase changed during a foreground fetch");
        source_prefill_ = source_prefill;
    }

    class Fetch {
    public:
        Fetch(ExpertSsdReadPhase & owner, bool direct)
            : owner_(owner), direct_(direct) {
            ++owner_.active_fetches_;
        }
        Fetch(const Fetch &) = delete;
        Fetch & operator=(const Fetch &) = delete;
        ~Fetch() { --owner_.active_fetches_; }
        bool direct() const { return direct_; }
    private:
        ExpertSsdReadPhase & owner_;
        const bool direct_;
    };

    Fetch begin_fetch(bool configured_direct) {
        const bool direct = effective_direct(configured_direct);
        if (source_prefill_) {
            if (direct) ++scopes_.source_direct;
            else ++scopes_.source_buffered;
        } else {
            if (direct) ++scopes_.non_source_direct;
            else ++scopes_.non_source_buffered;
        }
        return Fetch(*this, direct);
    }
    bool buffered_source_enabled() const { return buffered_source_; }
    const ScopeCounts & scope_counts() const { return scopes_; }
    void append_json(FILE * file, bool configured_direct) const {
        // Owner fetch scopes include RAM-hit/empty/remote early returns; these
        // counters must never be interpreted as physical SSD reads or bytes.
        std::fprintf(file, ",\"ssd_io_phase\":{\"buffered_source_option\":%s,"
            "\"configured_direct\":%s,\"current_source_prefill\":%s,\"effective_direct\":%s,"
            "\"counter_kind\":\"owner_fetch_scopes_not_actual_ssd_reads\","
            "\"source_direct_scopes\":%llu,\"source_buffered_scopes\":%llu,"
            "\"non_source_direct_scopes\":%llu,\"non_source_buffered_scopes\":%llu}",
            buffered_source_ ? "true" : "false", configured_direct ? "true" : "false",
            source_prefill_ ? "true" : "false", effective_direct(configured_direct) ? "true" : "false",
            static_cast<unsigned long long>(scopes_.source_direct),
            static_cast<unsigned long long>(scopes_.source_buffered),
            static_cast<unsigned long long>(scopes_.non_source_direct),
            static_cast<unsigned long long>(scopes_.non_source_buffered));
    }

private:
    static bool parse(const char * value) {
        if (!value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("QWEN_SSD_PREFILL_BUFFERED must be 0 or 1");
    }
    bool buffered_source_ = false;
    bool source_prefill_ = false;
    unsigned active_fetches_ = 0;
    ScopeCounts scopes_;
    bool effective_direct(bool configured_direct) const {
        return configured_direct && !(buffered_source_ && source_prefill_);
    }
};

} // namespace questwend
