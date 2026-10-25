#pragma once
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace expert_cost {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
inline Json distribution(std::vector<double> values) {
    if (values.empty()) return {{"samples", 0}};
    std::sort(values.begin(), values.end());
    auto percentile = [&](double quantile) {
        return values[std::min(values.size() - 1,
            (size_t) std::ceil(quantile * values.size()) - 1)];
    };
    const size_t middle = values.size() / 2;
    return {{"samples", values.size()}, {"min_ms", values.front()},
        {"max_ms", values.back()}, {"mean_ms", std::accumulate(values.begin(), values.end(), 0.0) / values.size()},
        {"median_ms", values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2},
        {"p95_ms", percentile(.95)}, {"p99_ms", percentile(.99)},
        {"percentile_method", "nearest rank; median averages central pair"},
        {"p99_tail_observations", values.size() / 100}};
}
}
