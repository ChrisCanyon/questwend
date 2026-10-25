#include "host_expert_cache.h"
#include "hybrid_schedule.h"
#include "expert_admission.h"
#include "activation_policy.h"

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#include <filesystem>

using questwend::HostExpertCache;

void check(bool value) { if (!value) throw std::runtime_error("host cache invariant failed"); }

int main() {
    try {
        HostExpertCache cache(12);
        auto put = [&](uint64_t key, uint8_t value) {
            return cache.put(key, 4, [&](void * destination) { std::memset(destination, value, 4); });
        };
        uint8_t output[8]{};
        check(put(1, 11) && put(2, 22) && put(3, 33));
        check(cache.stats().resident_bytes == 12);
        check(cache.contains_and_touch(3));
        check(cache.copy(3, output, 4) && output[0] == 33);
        check(cache.read(3, output, 4) && output[0] == 33);
        check(cache.read(3, output, 4) && output[3] == 33);
        check(cache.stats().resident_bytes == 12 && cache.stats().cpu_reads == 1);
        check(cache.take(2, output, 4) && output[0] == 22 && output[3] == 22);
        check(cache.stats().resident_bytes == 8); // exclusive promotion
        check(put(4, 44) && put(5, 55));
        check(!cache.take(1, output, 4)); // oldest demoted expert was evicted
        check(cache.take(3, output, 4) && output[0] == 33);
        bool wrong_size = false;
        try { cache.take(4, output, 8); } catch (const std::runtime_error &) { wrong_size = true; }
        check(wrong_size && cache.take(4, output, 4) && output[0] == 44);
        check(!cache.put(8, 13, [](void *) { throw std::runtime_error("must not read"); }));
        bool failed_read = false;
        try { cache.put(9, 4, [](void *) { throw std::runtime_error("read failure"); }); }
        catch (const std::runtime_error &) { failed_read = true; }
        check(failed_read && !cache.take(9, output, 4));
        cache.trim(0);
        check(cache.stats().resident_bytes == 0 && cache.stats().pooled_bytes == 0 && cache.stats().peak_bytes <= 12);
        check(cache.stats().reused_allocations > 0);

        check(put(10, 77));
        auto borrowed = cache.borrow(10, 4);
        cache.trim(0);
        check(cache.stats().retired_leased_bytes == 4);
        for (int key = 20; key < 40; ++key) check(put(key, 88));
        check(borrowed && borrowed[0] == 77 && borrowed[3] == 77);
        check(cache.stats().resident_bytes + cache.stats().pooled_bytes + cache.stats().retired_leased_bytes <= 12);
        borrowed.reset();
        check(cache.stats().retired_leased_bytes == 0);

        HostExpertCache fully_leased(4);
        check(fully_leased.put(1, 4, [](void * dst) { std::memset(dst, 9, 4); }));
        auto only_lease = fully_leased.borrow(1, 4);
        fully_leased.trim(0);
        check(!fully_leased.put(2, 4, [](void *) { throw std::runtime_error("must reject before allocating"); }));
        only_lease.reset();
        check(fully_leased.put(2, 4, [](void * dst) { std::memset(dst, 8, 4); }));

        HostExpertCache threaded(1024 * 64);
        std::atomic<bool> good{true};
        std::vector<std::thread> threads;
        for (uint64_t worker = 0; worker < 4; ++worker) {
            threads.emplace_back([&, worker] {
                for (uint64_t index = 0; index < 2000; ++index) {
                    const uint64_t key = worker * 10000 + index;
                    uint8_t data[64];
                    threaded.put(key, sizeof(data), [&](void * dst) { std::memset(dst, (int) worker, sizeof(data)); });
                    if (threaded.take(key, data, sizeof(data)))
                        for (auto value : data) if (value != worker) good = false;
                }
            });
        }
        for (auto & thread : threads) thread.join();
        check(good && threaded.stats().resident_bytes + threaded.stats().pooled_bytes <= 1024 * 64);
        using questwend::HybridExpertTask;
        using questwend::schedule_hybrid_experts;
        auto cpu_plan = schedule_hybrid_experts({{0, 1, true, false, .1, .1, 0},
                                                 {1, 1, false, true, .01, .1, .5}});
        check(cpu_plan.gpu == std::vector<int>{0} && cpu_plan.cpu == std::vector<int>{1});
        auto gpu_plan = schedule_hybrid_experts({{0, 1, true, false, .1, 1, 0},
                                                 {1, 1, false, true, .1, 1, .1}});
        check(gpu_plan.gpu == std::vector<int>({0, 1}) && gpu_plan.cpu.empty());
        auto balanced = schedule_hybrid_experts({{0, 1, false, true, .1, .2, .15},
                                                  {1, 1, false, true, .1, .2, .15},
                                                  {2, 1, false, true, .1, .2, .15},
                                                  {3, 1, false, true, .1, .2, .15}});
        check(balanced.gpu == std::vector<int>({0, 1}) && balanced.cpu == std::vector<int>({3, 2}));
        check(schedule_hybrid_experts({}).gpu.empty());
        auto barrier = schedule_hybrid_experts({{0, 1, true, false, 2, 0, 0},
                                               {1, 1, false, false, 1, 0, 3}},
                                               0, .5, questwend::HybridGpuTimeline::BatchBarrier);
        check(std::fabs(barrier.gpu_finish_ms - 6.5) < .00001);
        bool invalid_cost = false;
        try { schedule_hybrid_experts({{0, 1, false, true, -.1, 1, .1}}); }
        catch (const std::invalid_argument &) { invalid_cost = true; }
        check(invalid_cost);
        questwend::ExpertAdmission admission(32, 4);
        for (int repeat = 0; repeat < 20; ++repeat) admission.observe(1);
        admission.observe(2);
        check(!admission.admit(2, 1) && admission.admit(1, 2));
        admission.observe(3);
        check(!admission.admit(3, 2)); // preserve resident on equal frequency
        // A phase change must age the old hot expert even when the active set
        // is small. CPU/transient accesses use the same demand clock.
        for (int repeat = 0; repeat < 6000; ++repeat) admission.observe(2);
        check(admission.admit(2, 1) && admission.frequency(1) < 20);
        HostExpertCache scored(8);
        scored.set_priority([](uint64_t key) { return key == 1 ? 10.0 : 1.0; });
        for (int key = 1; key <= 3; ++key)
            check(scored.put(key, 4, [](void * dst) { std::memset(dst, 1, 4); }));
        check(scored.contains(1) && !scored.contains(2) && scored.contains(3));
        {
            HostExpertCache linear(32), indexed(32);
            double priorities[32]{};
            auto score = [&](uint64_t key) { return priorities[key]; };
            linear.set_priority(score);
            indexed.set_priority(score, true);
            uint32_t state = 12345;
            auto next = [&] { state = state * 1664525u + 1013904223u; return state; };
            for (int step = 0; step < 3000; ++step) {
                const uint64_t changed = next() % 32;
                priorities[changed] = next() % 8; // many ties exercise LRU ordering
                indexed.refresh_priority_range(changed, 1);
                const uint64_t key = next() % 32;
                if (step % 7 == 0) {
                    uint8_t left[4]{}, right[4]{};
                    check(linear.take(key, left, 4) == indexed.take(key, right, 4));
                    check(std::memcmp(left, right, 4) == 0);
                } else if (step % 5 == 0) {
                    check(linear.contains_and_touch(key) == indexed.contains_and_touch(key));
                } else {
                    auto fill = [&](void * dst) { std::memset(dst, (int) key, 4); };
                    check(linear.put(key, 4, fill) == indexed.put(key, 4, fill));
                }
                if (step % 127 == 0) { linear.trim(0); indexed.trim(0); }
                for (uint64_t candidate = 0; candidate < 32; ++candidate)
                    check(linear.contains(candidate) == indexed.contains(candidate));
            }
            check(indexed.stats().eviction_candidates == 0);
        }

        const auto temporary = std::filesystem::temp_directory_path();
        const auto stem = "qw-activation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        const auto trace_a = temporary / (stem + "-a.tsv");
        const auto trace_b = temporary / (stem + "-b.tsv");
        const auto manifest = temporary / (stem + "-manifest.txt");
        questwend::ActivationPolicy a(3, 4), b(3, 4), query(3, 4);
        const int32_t one = 1, two = 2, three = 3;
        a.observe(0, &one, 1); a.observe(1, &three, 1);
        b.observe(0, &two, 1); b.observe(1, &one, 1);
        a.save_trace(trace_a.string()); b.save_trace(trace_b.string());
        { std::ofstream listing(manifest); listing << trace_a.string() << '\n' << trace_b.string() << '\n'; }
        query.load_references(manifest.string());
        query.observe(0, &one, 1);
        check(query.closest_reference(0) == 0 && query.predict_next(0, 2) == std::vector<int>{3});
        check(query.priority(1) == 1.0 && query.priority(0) == 0);
        check(query.predict_next(2, 2).empty());
        questwend::ActivationPolicy other(3, 4);
        other.load_references(manifest.string()); other.observe(0, &two, 1);
        check(other.closest_reference(0) == 1 && other.predict_next(0, 1) == std::vector<int>{1});
        std::filesystem::remove(trace_a); std::filesystem::remove(trace_b); std::filesystem::remove(manifest);
        std::cout << "{\"state\":\"passed\",\"threaded_operations\":16000,\"exclusive_promotion\":true,\"bounded_eviction\":true,\"indexed_eviction_matches_linear\":true}" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
