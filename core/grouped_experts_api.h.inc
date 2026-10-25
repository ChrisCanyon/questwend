    // Isolated experiment. The parent retains attention, recurrent state,
    // routing, shared experts and sampling. Only selected routed FFNs leave.
    struct GroupedExpertRequest {
        int layer = 0, position = 0, token = 0;
        std::vector<int32_t> experts;
        std::vector<float> weights, input;
        std::vector<int32_t> selected_experts;
        std::vector<float> selected_weights;
        std::vector<uint8_t> locally_gpu_resident;
        std::vector<uint8_t> locally_ram_resident;
    };
    using GroupedExpertExecutor = std::function<std::vector<float>(const GroupedExpertRequest &)>;
    // Configure AFTER ordinary prefill. Empty executor/count0 is the matched
    // all-local segmented control. Both use the existing p_corr/HC pathway.
    // Fixed split selects the first count router entries, without renormalizing.
    void configure_grouped_experts(const std::vector<int> & layers, int count,
                                  GroupedExpertExecutor executor = {}, bool misses_only = false,
                                  bool ssd_misses_only = false);
