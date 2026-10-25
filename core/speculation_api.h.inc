    // Experimental greedy external drafting. Text-only; use_mtp must be false.
    // These probes preserve Flash's PLE convolution and n-gram state in addition
    // to ordinary attention/GDN state. They do not change production endpoints.
    struct VerificationProbe {
        std::vector<int32_t> next_tokens; // argmax after every input position
        int input_tokens = 0;
        int kept_tokens = 0;
        int accepted_drafts = 0;
        int32_t next_token = -1;
        bool batched = false;
        bool replayed_for_state = false;
        double prepare_ms = 0;
        double verify_ms = 0;
        double settle_ms = 0;
    };

    // keep_tokens=0..tokens.size(): retain precisely that many input tokens.
    // keep_tokens=-1: tokens[0] is an authoritative pending token; remaining
    // tokens are draft proposals. Keep the longest greedy-consistent prefix.
    VerificationProbe probe_verify_block(const std::vector<int32_t> & tokens,
                                          int keep_tokens);
    VerificationProbe verify_external_greedy(int32_t pending_token,
                                              const std::vector<int32_t> & drafts);

    // Probe-only snapshot format adds PLE, n-gram history, QSA indexer keys and
    // the current logits to save_state(). Keep at most one snapshot in RAM.
    void save_probe_state(const std::function<void(const void *, size_t)> & sink) const;
    void load_probe_state(const std::function<void(void *, size_t)> & source);
