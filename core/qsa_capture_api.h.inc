    // Isolated diagnostic: call immediately after a single cached scalar decode.
    // Binary tensor spans preserve ggml shapes/strides; callback must consume
    // bytes synchronously. No model or cache state is changed by export.
    struct QsaCaptureTensor {
        std::string name, dtype;
        std::vector<int64_t> shape;
        std::vector<uint64_t> strides;
        bool contiguous = false;
        int finite_bound = 0;
        float scale = 0;
    };
    void save_qsa_attention_capture(int layer,
        const std::function<void(const QsaCaptureTensor &, const void *, size_t)> & sink) const;
