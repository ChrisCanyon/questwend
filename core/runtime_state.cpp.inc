// Prompt-state serialization. Included after Runtime::Impl; payload bytes are unchanged.
namespace {
struct StateHeader {
    uint32_t magic;      // 'QSS3'
    int32_t  n_layer, n_ctx, n_embd_gqa;
    int32_t  n_past, mtp_past, n_toks, n_hidden, mrope_next;
};
// Bumped to QSS3 when the KV cache became F16: the per-row byte count halved,
// so a QSS2 slot file on disk would pass the header check and then be misread.
constexpr uint32_t STATE_MAGIC = 0x33535351;   // "QSS3" little-endian
}



size_t Runtime::state_bytes() const {
    const auto & hp = impl_->model.hparams();
    size_t n = sizeof(StateHeader);
    n += impl_->kv_toks.size() * sizeof(int32_t);
    n += impl_->mtp_hidden.size() * sizeof(float);
    for (int il = 0; il < (int) hp.n_layer; ++il) {
        if (impl_->k_cache[il]) {
            const int rows = il >= (int) hp.n_main() ? impl_->mtp_past : impl_->n_past;
            n += 2 * (size_t) rows * impl_->k_cache[il]->nb[1];
        } else {
            n += ggml_nbytes(impl_->conv_state[il]) + (impl_->ssm_state[il] ? ggml_nbytes(impl_->ssm_state[il]) : 0);
        }
    }
    return n;
}

void Runtime::save_state(const std::function<void(const void *, size_t)> & sink) const {
    const auto & hp = impl_->model.hparams();
    StateHeader h{};
    h.magic      = impl_->cfg.kv_cache_type == "q8_0" ? 0x34535351 : STATE_MAGIC; // QSS4 separates Q8 slot bytes
    h.n_layer    = (int32_t) hp.n_layer;
    h.n_ctx      = impl_->n_ctx;
    h.n_embd_gqa = (int32_t) (hp.n_head_kv * hp.n_embd_head);
    h.n_past     = impl_->n_past;
    h.mtp_past   = impl_->mtp_past;
    h.n_toks     = (int32_t) impl_->kv_toks.size();
    h.n_hidden   = (int32_t) impl_->mtp_hidden.size();
    h.mrope_next = impl_->mrope_next;
    sink(&h, sizeof(h));
    if (h.n_toks)   sink(impl_->kv_toks.data(),    h.n_toks   * sizeof(int32_t));
    if (h.n_hidden) sink(impl_->mtp_hidden.data(), h.n_hidden * sizeof(float));

    std::vector<uint8_t> buf;
    auto dump = [&](ggml_tensor * t, size_t nbytes) {
        if (nbytes == 0) return;
        buf.resize(nbytes);
        ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
        sink(buf.data(), nbytes);
    };
    for (int il = 0; il < h.n_layer; ++il) {
        if (impl_->k_cache[il]) {
            const int rows = il >= (int) hp.n_main() ? h.mtp_past : h.n_past;
            dump(impl_->k_cache[il], (size_t) rows * impl_->k_cache[il]->nb[1]);
            dump(impl_->v_cache[il], (size_t) rows * impl_->v_cache[il]->nb[1]);
        } else {
            dump(impl_->conv_state[il], ggml_nbytes(impl_->conv_state[il]));
            if (impl_->ssm_state[il]) dump(impl_->ssm_state[il], ggml_nbytes(impl_->ssm_state[il]));
        }
    }
}

void Runtime::load_state(const std::function<void(void *, size_t)> & src) {
    const auto & hp = impl_->model.hparams();
    StateHeader h{};
    src(&h, sizeof(h));
    const uint32_t expected_magic = impl_->cfg.kv_cache_type == "q8_0" ? 0x34535351 : STATE_MAGIC;
    if (h.magic != expected_magic ||
        h.n_layer != (int32_t) hp.n_layer || h.n_ctx != impl_->n_ctx ||
        h.n_embd_gqa != (int32_t) (hp.n_head_kv * hp.n_embd_head) ||
        h.n_past < 0 || h.n_past > impl_->n_ctx || h.mtp_past < 0 ||
        h.n_toks < 0 || h.n_hidden < 0)
        throw std::runtime_error("load_state: header mismatch (different model/n_ctx or corrupt slot)");

    std::vector<int32_t> toks(h.n_toks);
    if (h.n_toks) src(toks.data(), h.n_toks * sizeof(int32_t));
    std::vector<float> hidden(h.n_hidden);
    if (h.n_hidden) src(hidden.data(), h.n_hidden * sizeof(float));

    std::vector<uint8_t> buf;
    auto fill = [&](ggml_tensor * t, size_t nbytes) {
        if (nbytes == 0) return;
        buf.resize(nbytes);
        src(buf.data(), nbytes);
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    };
    for (int il = 0; il < h.n_layer; ++il) {
        if (impl_->k_cache[il]) {
            const int rows = il >= (int) hp.n_main() ? h.mtp_past : h.n_past;
            fill(impl_->k_cache[il], (size_t) rows * impl_->k_cache[il]->nb[1]);
            fill(impl_->v_cache[il], (size_t) rows * impl_->v_cache[il]->nb[1]);
        } else {
            fill(impl_->conv_state[il], ggml_nbytes(impl_->conv_state[il]));
            if (impl_->ssm_state[il]) fill(impl_->ssm_state[il], ggml_nbytes(impl_->ssm_state[il]));
        }
    }
    impl_->n_past     = h.n_past;
    impl_->mtp_past   = h.mtp_past;
    impl_->mrope_next = h.mrope_next;
    impl_->kv_toks    = std::move(toks);
    impl_->mtp_hidden = std::move(hidden);
    impl_->pk.clear();   // checkpoints belong to the replaced lineage
}

void Runtime::save_complete_state(
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

void Runtime::load_complete_state(
        const std::function<void(void *, size_t)> & source) {
    uint64_t magic = 0;
    source(&magic, sizeof(magic));
    if (magic != 0x3156424f525051ull)
        throw std::runtime_error("complete snapshot format mismatch");
    load_state(source);
    auto vector = [&](auto & values, size_t limit) {
        uint64_t count = 0;
        source(&count, sizeof(count));
        if (count > limit) throw std::runtime_error("invalid complete snapshot vector size");
        values.resize((size_t) count);
        if (count) source(values.data(), count * sizeof(values[0]));
    };
    vector(impl_->ngram_hist, impl_->n_ctx);
    vector(impl_->logits, impl_->model.vocab().tokens.size());
    std::vector<uint8_t> buffer;
    auto tensor = [&](ggml_tensor * value, size_t bytes) {
        uint64_t count = 0;
        source(&count, sizeof(count));
        if (count != bytes) throw std::runtime_error("complete snapshot tensor shape mismatch");
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
