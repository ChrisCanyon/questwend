// Included inside namespace questwend after Runtime::Impl and public wrappers.
// Deliberately separate from production HTTP handlers and MTP generation.

Runtime::VerificationProbe Runtime::probe_verify_block(
        const std::vector<int32_t> & tokens, int keep_tokens) {
    using Clock = std::chrono::steady_clock;
    auto milliseconds = [](Clock::duration value) {
        return std::chrono::duration<double, std::milli>(value).count();
    };
    auto & rt = *impl_;
    if (rt.cfg.use_mtp)
        throw std::runtime_error("external verification requires use_mtp=false");
    if (rt.cache_fast_enabled || rt.resident_decode)
        throw std::runtime_error("verification probe requires exact routing and no speculative cache fast path");
    if (tokens.empty() || tokens.size() > 64)
        throw std::runtime_error("verification probe requires 1..64 input tokens");
    if (keep_tokens < -1 || keep_tokens > (int) tokens.size())
        throw std::runtime_error("invalid verification retention length");
    if (rt.n_past != (int) rt.kv_toks.size() || rt.n_past + (int) tokens.size() > rt.n_ctx)
        throw std::runtime_error("verification position/context invariant failed");
    if (!rt.embd_ovr.empty())
        throw std::runtime_error("external verification is text-only");
    const int vocabulary = (int) rt.model.vocab().tokens.size();
    for (int32_t token : tokens)
        if (token < 0 || token >= vocabulary)
            throw std::runtime_error("verification token outside model vocabulary");

    VerificationProbe result;
    result.input_tokens = (int) tokens.size();
    const int position = rt.n_past;
    const int rope_position = rt.mrope_next;
    const auto previous_history = rt.ngram_hist;
    const auto previous_logits = rt.logits;
    const bool recurrent = rt.model.hparams().has_recurrent();
    const auto prepare_start = Clock::now();
    if (recurrent) rt.backup_states();

    // PLE is stateful even though HParams::has_recurrent() only describes
    // GDN/short-convolution layers. KV truncation alone cannot undo PLE.
    std::vector<std::vector<uint8_t>> ple_before(rt.ple_conv_state.size());
    for (size_t layer = 0; layer < rt.ple_conv_state.size(); ++layer) {
        auto * tensor = rt.ple_conv_state[layer];
        if (!tensor) continue;
        ple_before[layer].resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, ple_before[layer].data(), 0,
                                ple_before[layer].size());
    }
    const bool has_ple_state = std::any_of(ple_before.begin(), ple_before.end(),
            [](const std::vector<uint8_t> & bytes) { return !bytes.empty(); });
    result.prepare_ms = milliseconds(Clock::now() - prepare_start);

    auto restore_prefix = [&]() {
        if (recurrent) rt.restore_states();
        for (size_t layer = 0; layer < ple_before.size(); ++layer)
            if (!ple_before[layer].empty())
                ggml_backend_tensor_set(rt.ple_conv_state[layer], ple_before[layer].data(),
                                        0, ple_before[layer].size());
        rt.ngram_hist = previous_history;
        rt.n_past = position;
        rt.mrope_next = rope_position;
        rt.logits = previous_logits;
    };

    try {
        const auto verify_start = Clock::now();
        rt.decode_verify(tokens);
        result.verify_ms = milliseconds(Clock::now() - verify_start);
        result.batched = !rt.ecache || rt.v_from_batch;
        result.next_tokens = rt.vA;
        if (result.next_tokens.size() != tokens.size())
            throw std::runtime_error("verification returned the wrong number of positions");

        if (keep_tokens == -1) {
            keep_tokens = 1;
            while (keep_tokens < (int) tokens.size() &&
                   result.next_tokens[keep_tokens - 1] == tokens[keep_tokens])
                ++keep_tokens;
        }
        result.kept_tokens = keep_tokens;
        result.accepted_drafts = std::max(0, keep_tokens - 1);
        if (keep_tokens > 0) result.next_token = result.next_tokens[keep_tokens - 1];

        const auto settle_start = Clock::now();
        if (keep_tokens == 0) {
            restore_prefix();
        } else if (keep_tokens != (int) tokens.size()) {
            if (has_ple_state) {
                // Correctness-first fallback: restore and replay only confirmed
                // tokens. Per-position PLE checkpoints are a future optimization.
                restore_prefix();
                rt.decode(std::vector<int32_t>(tokens.begin(), tokens.begin() + keep_tokens));
                result.replayed_for_state = true;
            } else if (recurrent && (!rt.ecache || rt.v_from_batch)) {
                rt.restore_ckpt(keep_tokens - 1);
            } else if (recurrent) {
                restore_prefix();
                rt.decode(std::vector<int32_t>(tokens.begin(), tokens.begin() + keep_tokens));
                result.replayed_for_state = true;
            }
        }
        rt.n_past = position + keep_tokens;
        rt.mrope_next = rope_position + keep_tokens;
        rt.kv_toks.insert(rt.kv_toks.end(), tokens.begin(), tokens.begin() + keep_tokens);
        result.settle_ms = milliseconds(Clock::now() - settle_start);
        return result;
    } catch (...) {
        restore_prefix();
        rt.kv_toks.resize(position);
        throw;
    }
}

Runtime::VerificationProbe Runtime::verify_external_greedy(
        int32_t pending_token, const std::vector<int32_t> & drafts) {
    std::vector<int32_t> tokens;
    tokens.reserve(1 + drafts.size());
    tokens.push_back(pending_token);
    tokens.insert(tokens.end(), drafts.begin(), drafts.end());
    return probe_verify_block(tokens, -1);
}

void Runtime::save_probe_state(
        const std::function<void(const void *, size_t)> & sink) const {
    const uint64_t magic = 0x3156424f525051ull; // private QPROBV1 marker
    sink(&magic, sizeof(magic));
    save_state(sink);
    auto vector = [&](const auto & values) {
        const uint64_t count = values.size();
        sink(&count, sizeof(count));
        if (count) sink(values.data(), count * sizeof(values[0]));
    };
    vector(impl_->ngram_hist);
    vector(impl_->logits);
    std::vector<uint8_t> buffer;
    auto tensor = [&](ggml_tensor * value, size_t bytes) {
        const uint64_t count = bytes;
        sink(&count, sizeof(count));
        if (!count) return;
        buffer.resize(bytes);
        ggml_backend_tensor_get(value, buffer.data(), 0, bytes);
        sink(buffer.data(), bytes);
    };
    for (auto * value : impl_->ple_conv_state)
        tensor(value, value ? ggml_nbytes(value) : 0);
    for (auto * value : impl_->idx_k_cache)
        tensor(value, value ? (size_t) impl_->n_past * value->nb[1] : 0);
}

void Runtime::load_probe_state(
        const std::function<void(void *, size_t)> & source) {
    uint64_t magic = 0;
    source(&magic, sizeof(magic));
    if (magic != 0x3156424f525051ull)
        throw std::runtime_error("probe snapshot format mismatch");
    load_state(source);
    auto vector = [&](auto & values, size_t limit) {
        uint64_t count = 0;
        source(&count, sizeof(count));
        if (count > limit) throw std::runtime_error("invalid probe snapshot vector size");
        values.resize((size_t) count);
        if (count) source(values.data(), count * sizeof(values[0]));
    };
    vector(impl_->ngram_hist, impl_->n_ctx);
    vector(impl_->logits, impl_->model.vocab().tokens.size());
    std::vector<uint8_t> buffer;
    auto tensor = [&](ggml_tensor * value, size_t bytes) {
        uint64_t count = 0;
        source(&count, sizeof(count));
        if (count != bytes) throw std::runtime_error("probe tensor shape mismatch");
        if (!count) return;
        buffer.resize(bytes);
        source(buffer.data(), bytes);
        ggml_backend_tensor_set(value, buffer.data(), 0, bytes);
    };
    for (auto * value : impl_->ple_conv_state)
        tensor(value, value ? ggml_nbytes(value) : 0);
    for (auto * value : impl_->idx_k_cache)
        tensor(value, value ? (size_t) impl_->n_past * value->nb[1] : 0);
}
