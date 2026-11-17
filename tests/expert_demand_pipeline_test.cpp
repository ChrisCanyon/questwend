// Native CPU-only checks for ordering, immutable lease lifetime and cleanup.
#include "expert_io_workers.h"
#include "expert_readiness.h"
#include "host_expert_cache.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace questwend;

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void ordering_and_early_progress() {
    PersistentExpertIO pool(4);
    ExpertReadiness ready(32);
    std::vector<int> values(32, -1), writes(32, 0), owner_order;
    std::mutex mutex;
    std::condition_variable changed;
    bool owner_consumed_first = false;
    const auto owner_thread = std::this_thread::get_id();
    pool.run_with_owner_progress(4, [&](int worker, ExpertIOFiles &) {
        for (int job = worker; job < 32; job += 4) {
            try {
                if (job == 1) {
                    std::unique_lock<std::mutex> lock(mutex);
                    if (!changed.wait_for(lock, std::chrono::seconds(2), [&] { return owner_consumed_first; }))
                        throw std::runtime_error("Owner did not run until after bulk reader barrier");
                }
                require(std::this_thread::get_id() != owner_thread, "Read callback ran on owner");
                values[job] = job * 7;
                ++writes[job];
                ready.publish(job);
            } catch (...) { ready.fail(std::current_exception()); throw; }
        }
    }, [&] {
        for (int job = 0; job < 32; ++job) {
            ready.wait(job);
            require(std::this_thread::get_id() == owner_thread, "Owner callback changed thread");
            require(values[job] == job * 7 && writes[job] == 1, "Read not visible or span written twice");
            owner_order.push_back(job);
            if (job == 0) {
                std::lock_guard<std::mutex> lock(mutex);
                owner_consumed_first = true;
                changed.notify_all();
            }
        }
    });
    for (int job = 0; job < 32; ++job)
        require(owner_order[job] == job, "Admission/upload order changed");
}

static void frozen_lease_survives_owner_eviction() {
    HostExpertCache cache(32);
    require(cache.put(10, 16, [](void * data) { std::memset(data, 0x51, 16); }), "Initial put");
    auto lease = cache.transfer_lease(10, 16, false);
    require(lease != nullptr, "Read lease missing");
    cache.trim(0); // Owner-only mutation while a reader could still use bytes.
    require(!cache.contains(10), "Eviction did not remove key");
    require(cache.stats().retired_leased_bytes == 16, "Lease not budget-accounted");
    for (int i = 0; i < 16; ++i) require(lease[i] == 0x51, "Eviction recycled live bytes");
    require(cache.put(11, 16, [](void * data) { std::memset(data, 0x72, 16); }), "Non-overlapping new put");
    for (int i = 0; i < 16; ++i) require(lease[i] == 0x51, "Admission overwrote frozen bytes");
    lease.reset();
    auto taken = cache.transfer_lease(11, 16, true);
    require(taken && !cache.contains(11), "Take lease retained managed RAM ownership");
    for (int i = 0; i < 16; ++i) require(taken[i] == 0x72, "Take lease bytes changed");
    require(!cache.transfer_lease(12, 16, true), "Missing source returned a lease");
}

static void worker_error_wakes_owner_and_pool_reuses() {
    PersistentExpertIO pool(2);
    ExpertReadiness ready(2);
    std::atomic<int> readers_finished{0};
    bool failed = false;
    try {
        pool.run_with_owner_progress(2, [&](int worker, ExpertIOFiles &) {
            try {
                if (worker == 0) throw std::runtime_error("Injected SSD read failure");
                ready.publish(1);
            } catch (...) { ready.fail(std::current_exception()); ++readers_finished; throw; }
            ++readers_finished;
        }, [&] { ready.wait(0); });
    } catch (const std::runtime_error &) { failed = true; }
    require(failed && readers_finished == 2, "Read failure leaked a running writer");
    std::atomic<int> reuse{0};
    pool.run(2, [&](int, ExpertIOFiles &) { ++reuse; });
    require(reuse == 2, "Pool unusable after failed pipeline");
}

static void owner_error_still_drains_writers() {
    PersistentExpertIO pool(2);
    std::atomic<int> finished{0};
    bool failed = false;
    try {
        pool.run_with_owner_progress(2, [&](int, ExpertIOFiles &) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ++finished;
        }, [] { throw std::runtime_error("Injected admission/upload failure"); });
    } catch (const std::runtime_error &) { failed = true; }
    require(failed && finished == 2, "Owner failure returned with staging writers active");
}

static void retirement_allocation_failure_preserves_residency() {
    for (bool indexed : {false, true}) {
        HostExpertCache cache(32);
        require(cache.put(1, 16, [](void * data) { std::memset(data, 0x19, 16); }), "Fault fixture1");
        require(cache.put(2, 16, [](void * data) { std::memset(data, 0x29, 16); }), "Fault fixture2");
        if (indexed) cache.set_priority([](uint64_t key) { return static_cast<double>(key); }, true);
        const auto before = cache.stats();
        cache.fail_erase_allocation_for_test(1);
        bool failed = false;
        try { cache.transfer_lease(1, 16, true); }
        catch (const std::bad_alloc &) { failed = true; }
        const auto after = cache.stats();
        require(failed, "Required retirement allocation failure was swallowed");
        require(cache.contains(1) && cache.contains(2), "Failed retirement removed a live key");
        require(after.resident_bytes == before.resident_bytes && after.retired_leased_bytes == 0,
                "Failed retirement changed residency/retirement accounting");
        require(after.hits == before.hits && after.hit_bytes == before.hit_bytes,
                "Failed transfer recorded a successful hit");
        // Both LRU and indexed eviction must still find the ORIGINAL first key.
        require(cache.put(3, 16, [](void * data) { std::memset(data, 0x39, 16); }), "Post-failure put");
        require(!cache.contains(1) && cache.contains(2) && cache.contains(3),
                "Failed retirement damaged LRU order or the priority index");
        auto lease = cache.transfer_lease(2, 16, true);
        require(lease && lease[0] == 0x29 && !cache.contains(2), "Retirement retry failed");
        require(cache.stats().retired_leased_bytes == 16, "Successful retry lease is not bounded");
    }
}

static void optional_spare_allocation_failure_discards_safely() {
    for (int fault : {2, 3}) {
        HostExpertCache cache(16);
        require(cache.put(1, 16, [](void * data) { std::memset(data, 0x4b, 16); }), "Spare fixture");
        cache.fail_erase_allocation_for_test(fault);
        uint8_t copied[16]{};
        require(cache.take(1, copied, 16), "Optional spare allocation prevented take");
        require(!cache.contains(1), "Failed spare allocation retained managed ownership");
        const auto after = cache.stats();
        require(after.resident_bytes == 0 && after.pooled_bytes == 0 && after.retired_leased_bytes == 0,
                "Failed spare allocation corrupted byte accounting");
        for (uint8_t byte : copied) require(byte == 0x4b, "Take bytes lost during spare failure");
        require(cache.put(2, 16, [](void * data) { std::memset(data, 0x5c, 16); }), "Cache unusable after spare failure");
        auto lease = cache.transfer_lease(2, 16, false);
        require(lease && lease[0] == 0x5c, "New cache data changed after spare failure");
    }
}

static void failed_leased_eviction_does_not_commit_counters() {
    for (bool indexed : {false, true}) {
        HostExpertCache cache(32);
        require(cache.put(1, 16, [](void * data) { std::memset(data, 1, 16); }), "Eviction fixture1");
        require(cache.put(2, 16, [](void * data) { std::memset(data, 2, 16); }), "Eviction fixture2");
        if (indexed) cache.set_priority([](uint64_t key) { return static_cast<double>(key); }, true);
        auto pinned = cache.borrow(1, 16);
        cache.contains_and_touch(2); // Keep key1 first for both eviction policies.
        const auto before = cache.stats();
        cache.fail_erase_allocation_for_test(1);
        bool failed = false;
        try { cache.put(3, 16, [](void * data) { std::memset(data, 3, 16); }); }
        catch (const std::bad_alloc &) { failed = true; }
        const auto after = cache.stats();
        require(failed && cache.contains(1) && cache.contains(2) && !cache.contains(3),
                "Failed leased eviction changed key ownership");
        require(after.resident_bytes == before.resident_bytes && after.evictions == before.evictions &&
                after.evicted_bytes == before.evicted_bytes && after.retired_leased_bytes == 0,
                "Failed leased eviction committed accounting");
        require(pinned && pinned[0] == 1, "Failed leased eviction changed frozen bytes");
        pinned.reset();
        require(cache.put(3, 16, [](void * data) { std::memset(data, 3, 16); }), "Eviction retry failed");
        require(!cache.contains(1) && cache.contains(2) && cache.contains(3), "Eviction retry policy changed");
    }
}

int main() {
    try {
        ordering_and_early_progress();
        frozen_lease_survives_owner_eviction();
        worker_error_wakes_owner_and_pool_reuses();
        owner_error_still_drains_writers();
        retirement_allocation_failure_preserves_residency();
        optional_spare_allocation_failure_discards_safely();
        failed_leased_eviction_does_not_commit_counters();
        ExpertReadiness ready(1);
        ready.publish(0);
        bool duplicate = false;
        try { ready.publish(0); } catch (const std::logic_error &) { duplicate = true; }
        require(duplicate, "Duplicate completion silently accepted");
        std::cout << "expert demand pipeline ownership checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
