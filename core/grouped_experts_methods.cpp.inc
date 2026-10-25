void Runtime::configure_grouped_experts(const std::vector<int> & layers, int count,
                                       GroupedExpertExecutor executor, bool misses_only, bool ssd_misses_only) {
    auto & runtime = *impl_;
    const auto & hp = runtime.model.hparams();
    if (!runtime.ecache || runtime.multi_gpu() || runtime.block_worker() ||
        runtime.block_executor || runtime.block_observer || count < 0 ||
        count > (int) hp.n_expert_used || (count > 0 && !executor))
        throw std::runtime_error("Grouped gate needs one GPU, an expert cache, and a valid executor/count");
    for (int layer : layers) if (layer < 0 || layer >= (int) hp.n_main() || !hp.is_moe_layer(layer))
        throw std::runtime_error("Grouped expert layer is invalid");
    // Reuse the existing segmented split machinery without enabling a local
    // CPU execution lane, automatic HSS calibration, refill or weight borrowing.
    // Its zero correction follows precisely the same HC combine in control.
    runtime.cpu_miss = true;
    runtime.bounded_cpu_miss = true; runtime.bounded_cpu_max = 0;
    runtime.borrow_cpu_weights = false;
    runtime.overlap_bounded_cpu = false; runtime.overlap_cpu_io = false;
    runtime.hss_enabled = false; runtime.cm_refill = 0;
    runtime.grouped_expert_layers = layers;
    runtime.grouped_expert_layer_mask.assign(hp.n_main(), 0);
    for (int layer : layers) runtime.grouped_expert_layer_mask[layer] = 1;
    runtime.grouped_expert_count = count;
    runtime.grouped_expert_misses_only = misses_only;
    runtime.grouped_expert_ssd_misses_only = ssd_misses_only;
    runtime.grouped_expert_executor = std::move(executor);
    if (!runtime.hy_pin) {
        const size_t nu = hp.n_expert_used, ne = hp.n_embd;
        const size_t bytes = (nu * 4 + nu * 3 + nu + ne * 2) * 4 +
            (size_t) runtime.n_ctx * 2 + 4 * 4 + 8 + 512;
        auto device = ggml_backend_get_device(runtime.backend);
        auto pinned = device ? ggml_backend_dev_host_buffer_type(device) : nullptr;
        if (!pinned) throw std::runtime_error("Grouped gate needs CUDA host staging");
        runtime.hy_pin = ggml_backend_buft_alloc_buffer(pinned, bytes);
        if (!runtime.hy_pin) throw std::runtime_error("Grouped gate staging allocation failed");
        char * p = static_cast<char *>(ggml_backend_buffer_get_base(runtime.hy_pin));
        runtime.pin_sel = reinterpret_cast<int32_t *>(p); p += nu * 4;
        runtime.pin_ids = reinterpret_cast<int32_t *>(p); p += nu * 3 * 4;
        runtime.pin_w = reinterpret_cast<float *>(p); p += nu * 4;
        runtime.pin_gw = reinterpret_cast<float *>(p); p += nu * 4;
        runtime.pin_x = reinterpret_cast<float *>(p); p += ne * 4;
        runtime.pin_corr = reinterpret_cast<float *>(p); p += ne * 4;
        runtime.pin_kvidx = reinterpret_cast<int64_t *>((reinterpret_cast<uintptr_t>(p) + 7) & ~uintptr_t(7));
        p = reinterpret_cast<char *>(runtime.pin_kvidx + 1);
        runtime.pin_pos = reinterpret_cast<int32_t *>(p); p += 4 * 4;
        runtime.pin_mask = reinterpret_cast<ggml_fp16_t *>(p);
    }
}
