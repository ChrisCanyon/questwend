    std::vector<int> grouped_expert_layers;
    std::vector<uint8_t> grouped_expert_layer_mask;
    int grouped_expert_count = 0;
    bool grouped_expert_misses_only = false;
    bool grouped_expert_ssd_misses_only = false;
    Runtime::GroupedExpertExecutor grouped_expert_executor;
