// Adapted from HybriMoE HSS, PKU-SEC-Lab/HybriMoE, Apache License 2.0.
// Source commit 508a85b9a43dff37124370d8816e411e74827803,
// ktransformers/operators/experts.py:1136-1182.
// Changes: C++ port; caller-supplied measured costs; explicit CPU eligibility
// for exclusive tiers. Fixed-GPU tasks establish the GPU/copy lane backlog.
// Retain third_party/hybrimoe/LICENSE on distribution.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace questwend {

struct HybridExpertTask {
    int id = -1;
    int token_rows = 1;
    bool gpu_resident = false;
    bool cpu_eligible = false;
    double gpu_ms = 0;
    double cpu_ms = 0;
    double transfer_ms = 0;
};

struct HybridExpertPlan {
    std::vector<int> gpu, cpu;
    double gpu_finish_ms = 0, cpu_finish_ms = 0, copy_finish_ms = 0;
};

enum class HybridGpuTimeline { Pipelined, BatchBarrier };

inline HybridExpertPlan schedule_hybrid_experts(std::vector<HybridExpertTask> tasks,
                                               double cpu_startup_ms = 0,
                                               double gpu_startup_ms = 0,
                                               HybridGpuTimeline timeline = HybridGpuTimeline::Pipelined) {
    HybridExpertPlan plan;
    double gpu_compute_ms = 0;
    auto gpu_completion = [&](const HybridExpertTask & task) {
        if (timeline == HybridGpuTimeline::BatchBarrier)
            return plan.copy_finish_ms + (task.gpu_resident ? 0 : task.transfer_ms) +
                   gpu_compute_ms + task.gpu_ms + gpu_startup_ms;
        const double ready = task.gpu_resident ? 0 : plan.copy_finish_ms + task.transfer_ms;
        return std::max(plan.gpu_finish_ms, ready) + task.gpu_ms +
               (plan.gpu.empty() ? gpu_startup_ms : 0);
    };
    auto gpu_assign = [&](const HybridExpertTask & task) {
        plan.gpu_finish_ms = gpu_completion(task);
        if (!task.gpu_resident) plan.copy_finish_ms += task.transfer_ms;
        gpu_compute_ms += task.gpu_ms;
        plan.gpu.push_back(task.id);
    };
    for (const auto & task : tasks) {
        if (task.token_rows < 1 || task.gpu_ms < 0 || task.cpu_ms < 0 || task.transfer_ms < 0 ||
            !std::isfinite(task.gpu_ms + task.cpu_ms + task.transfer_ms))
            throw std::invalid_argument("Invalid hybrid expert cost");
    }
    std::stable_sort(tasks.begin(), tasks.end(), [](const auto & left, const auto & right) {
        if (left.gpu_resident != right.gpu_resident) return left.gpu_resident;
        return left.token_rows > right.token_rows;
    });
    std::vector<HybridExpertTask> flexible;
    for (const auto & task : tasks) {
        if (task.cpu_eligible) flexible.push_back(task);
        else gpu_assign(task);
    }
    // HSS compares the GPU-front candidate with the CPU-back candidate and
    // extends whichever lane would finish first. Transfer and GPU clocks are
    // separate so already queued copies can overlap prior GPU computation.
    size_t front = 0, back = flexible.size();
    while (front < back) {
        const auto & gpu = flexible[front];
        const auto & cpu = flexible[back - 1];
        // This runtime uploads the complete GPU subset before one graph launch.
        // BatchBarrier includes that barrier instead of assuming expert-by-expert
        // transfer/compute overlap from the original HybriMoE execution engine.
        const double gpu_finish = gpu_completion(gpu);
        const double cpu_finish = plan.cpu_finish_ms + cpu.cpu_ms +
                                  (plan.cpu.empty() ? cpu_startup_ms : 0);
        if (gpu_finish <= cpu_finish) {
            gpu_assign(gpu);
            ++front;
        } else {
            plan.cpu.push_back(cpu.id);
            plan.cpu_finish_ms = cpu_finish;
            --back;
        }
    }
    return plan;
}

} // namespace questwend
