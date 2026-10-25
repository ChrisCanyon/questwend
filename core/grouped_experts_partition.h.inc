            // Prototype fixed split. Routing order, coefficients and activation
            // remain those produced by this actual model layer/token. No CPU
            // fallback participates: configure_grouped_experts sets its limit0.
            if (grouped_expert_executor && grouped_expert_layer_mask.at(il)) {
                if (!cids.empty() || gpu_ids.size() != (size_t) n_used)
                    throw std::runtime_error("Grouped gate requires an exact all-local selection before splitting");
                std::vector<int32_t> retained_ids;
                std::vector<float> retained_weights;
                for (size_t index = 0; index < gpu_ids.size(); ++index) {
                    const int expert = gpu_ids[index];
                    const bool resident = ec->resident(ExpertCache::GATE, il, expert) &&
                        ec->resident(ExpertCache::UP, il, expert) && ec->resident(ExpertCache::DOWN, il, expert);
                    const bool ram_resident = ec->host_resident(il, expert, false);
                    const bool eligible = (!grouped_expert_misses_only || !resident) &&
                        (!grouped_expert_ssd_misses_only || (!resident && !ram_resident));
                    if ((int) cids.size() < grouped_expert_count && eligible) {
                        cids.push_back(expert); cw.push_back(gpu_weights[index]);
                    } else {
                        retained_ids.push_back(expert); retained_weights.push_back(gpu_weights[index]);
                    }
                }
                gpu_ids = std::move(retained_ids); gpu_weights = std::move(retained_weights);
            }
