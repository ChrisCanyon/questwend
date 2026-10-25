            if (grouped_expert_executor && !cids.empty()) {
                Runtime::GroupedExpertRequest request;
                request.layer = il; request.position = n_past; request.token = token;
                request.experts = cids; request.weights = cw;
                request.selected_experts.assign(pin_sel, pin_sel + n_used);
                request.selected_weights.assign(pin_w, pin_w + n_used);
                for (int expert : request.selected_experts) {
                    request.locally_gpu_resident.push_back(
                        ec->resident(ExpertCache::GATE, il, expert) && ec->resident(ExpertCache::UP, il, expert) &&
                        ec->resident(ExpertCache::DOWN, il, expert));
                    request.locally_ram_resident.push_back(ec->host_resident(il, expert, false));
                }
                request.input.assign(pin_x, pin_x + n_embd);
                const auto executor = grouped_expert_executor;
                cpu_future = std::async(std::launch::async, [request = std::move(request), executor, n_embd]() {
                    const auto started = std::chrono::steady_clock::now();
                    auto output = executor(request);
                    if (output.size() != (size_t) n_embd)
                        throw std::runtime_error("Grouped expert response width mismatch; reset required");
                    for (float value : output) if (!std::isfinite(value))
                        throw std::runtime_error("Grouped expert response is nonfinite; reset required");
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                    return std::make_pair(std::move(output), ms);
                });
            }
