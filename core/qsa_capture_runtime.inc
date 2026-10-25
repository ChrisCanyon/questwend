void Runtime::save_qsa_attention_capture(int layer,
        const std::function<void(const QsaCaptureTensor &, const void *, size_t)> & sink) const {
    const auto & rt = *impl_;
    const char * enabled = getenv("QWEN_QSA_CAPTURE_LAYER");
    if (!enabled || std::strcmp(enabled, "3") != 0 || layer != 3 || !sink ||
        rt.n_past < 1 || rt.n_past > 65536 ||
        (size_t) layer >= rt.hy_a.size() || !rt.hy_a[layer].gf)
        throw std::runtime_error("QSA capture requires enabled layer3 and a completed cached scalar decode <=65536tokens");
    auto * graph = rt.hy_a[layer].gf;
    ggml_backend_synchronize(rt.be_of(layer));
    std::vector<uint8_t> bytes;
    for (const char * name : {"q", "k", "v", "mask", "dense", "sparse", "original"}) {
        const auto tensor_name = std::string("qsa_capture_") + name + "_3";
        auto * tensor = ggml_graph_get_tensor(graph, tensor_name.c_str());
        if (!tensor || !(tensor->view_src ? tensor->view_src->buffer : tensor->buffer) ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16))
            throw std::runtime_error("Missing/unsupported retained QSA capture tensor " + tensor_name);
        const size_t size = ggml_nbytes(tensor);
        if (size == 0 || size > 128ull * 1024 * 1024)
            throw std::runtime_error("QSA capture tensor exceeds diagnostic byte bound");
        QsaCaptureTensor metadata;
        metadata.name = name;
        metadata.dtype = tensor->type == GGML_TYPE_F32 ? "float32" : "float16";
        for (int axis = 0; axis < 4; ++axis) {
            metadata.shape.push_back(tensor->ne[axis]);
            metadata.strides.push_back(tensor->nb[axis]);
        }
        metadata.contiguous = ggml_is_contiguous(tensor);
        if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            std::memcpy(&metadata.finite_bound, tensor->op_params + 4, sizeof(metadata.finite_bound));
            std::memcpy(&metadata.scale, tensor->op_params, sizeof(float));
        }
        bytes.resize(size);
        ggml_backend_tensor_get(tensor, bytes.data(), 0, size);
        sink(metadata, bytes.data(), size);
    }
}
