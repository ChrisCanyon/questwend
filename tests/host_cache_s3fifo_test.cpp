#include "host_expert_cache.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using questwend::HostExpertCache;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Action>
static void expect_throw(Action action, const char * message) {
    bool thrown = false;
    try { action(); } catch (const std::exception &) { thrown = true; }
    check(thrown, message);
}

static bool insert(HostExpertCache & cache, uint64_t key, size_t bytes = 8) {
    return cache.put(key, bytes, [=](void * destination) {
        std::memset(destination, static_cast<int>(key & 255), bytes);
    });
}

static void check_accounting(HostExpertCache & cache) {
    check(cache.invariant_for_test(), "Payload/list/policy invariant failed");
    const auto stats = cache.stats();
    check(stats.resident_bytes + stats.retired_leased_bytes + stats.pooled_bytes <= stats.capacity_bytes,
        "Physical resident/retired/spare allocation cap exceeded");
    if (std::string(stats.replacement_policy) == "s3fifo-role-slabs")
        check(stats.s3_small_bytes + stats.s3_main_bytes == stats.resident_bytes,
            "Policy bytes are not authoritative resident bytes");
}

static void configure(HostExpertCache & cache, bool s3) {
    if (s3) cache.enable_s3fifo_role_slabs(512, 8);
}

static void successful_reuse_only() {
    HostExpertCache cache(1000);
    configure(cache, true);
    insert(cache, 7);
    check(cache.s3_frequency_for_test(7) == 0, "Admission must not count as a hit");
    for (int i = 0; i < 100; ++i) {
        check(cache.contains(7), "Contains failed");
        check(cache.contains_and_touch(7), "Eligibility probe failed");
    }
    check(cache.s3_frequency_for_test(7) == 0, "Eligibility probes inflated reuse frequency");
    unsigned char output[8];
    check(cache.read(7, output, sizeof(output)), "Read failed");
    check(cache.s3_frequency_for_test(7) == 1, "Read did not record reuse");
    check(cache.copy(7, output, sizeof(output)), "CPU copy failed");
    check(cache.s3_frequency_for_test(7) == 2, "CPU copy did not record reuse");
    auto borrowed = cache.borrow(7, 8);
    check(cache.s3_frequency_for_test(7) == 3, "Borrow did not record reuse");
    auto retained = cache.transfer_lease(7, 8, false);
    check(cache.s3_frequency_for_test(7) == 4, "Retained transfer did not record reuse");
    auto taken = cache.transfer_lease(7, 8, true);
    check(!cache.contains(7), "Taken transfer kept managed RAM ownership");
    check(cache.s3_frequency_for_test(7) == 0, "External removal retained resident metadata");
    check(taken[0] == 7 && borrowed[7] == 7 && retained[3] == 7, "Immutable leases changed");
    check(cache.stats().retired_leased_bytes == 8, "Retired weight was not charged");
    check(cache.stats().s3_ghost_entries == 0, "GPU transfer manufactured a capacity ghost");
    check_accounting(cache);
}

static void publication_failures(bool s3) {
    for (int point = 1; point <= 2; ++point) {
        HostExpertCache cache(1000);
        configure(cache, s3);
        insert(cache, 1);
        cache.fail_publish_allocation_for_test(point);
        expect_throw([&] { insert(cache, 2); }, "Publication fault was not raised");
        check(cache.contains(1) && !cache.contains(2), "Failed publication exposed a new key");
        check(cache.stats().admissions == 1 && cache.stats().resident_bytes == 8,
            "Failed publication changed committed admission accounting");
        check_accounting(cache);
        check(insert(cache, 2), "Admission did not recover after publication failure");
        check_accounting(cache);
    }
    HostExpertCache cache(1000);
    configure(cache, s3);
    expect_throw([&] {
        cache.put(3, 8, [](void *) { throw std::runtime_error("Injected private fill failure"); });
    }, "Fill fault was not raised");
    check(!cache.contains(3) && cache.stats().resident_bytes == 0, "Failed fill was published");
    check_accounting(cache);

    // Indexed default policy has one additional allocating publication stage.
    if (!s3) {
        cache.set_priority([](uint64_t key) { return static_cast<double>(key); }, true);
        cache.fail_publish_allocation_for_test(3);
        expect_throw([&] { insert(cache, 4); }, "Rank publication fault was not raised");
        check(!cache.contains(4), "Failed rank insertion exposed a key");
        check_accounting(cache);
        insert(cache, 4);
        check_accounting(cache);
    }
}

static void retirement_failure(bool s3) {
    HostExpertCache cache(1000);
    configure(cache, s3);
    insert(cache, 0);
    auto pinned = cache.borrow(0, 8); // One hit leaves it a cold small-queue victim.
    for (uint64_t key = 1; key < 125; ++key) insert(cache, key);
    const auto before = cache.stats();
    cache.fail_erase_allocation_for_test(1);
    expect_throw([&] { insert(cache, 125); }, "Required retirement fault was not raised");
    check(cache.contains(0) && !cache.contains(125), "Failed retirement changed key ownership");
    check(cache.stats().resident_bytes == before.resident_bytes, "Failed retirement changed resident accounting");
    check(cache.stats().evictions == before.evictions, "Failed retirement counted an eviction");
    check_accounting(cache);
    check(pinned[0] == 0, "Pinned bytes changed after aborted capacity eviction");
    // With the victim still pinned, make room elsewhere while charging its
    // retired owner. This requires a second capacity victim, not overcommit.
    check(insert(cache, 125), "Admission did not recover after retirement failure");
    check_accounting(cache);
    check(pinned[7] == 0, "Evicted lease bytes changed");
}

static void optional_spare_failures(bool s3) {
    for (int point : {2, 3}) {
        HostExpertCache cache(1000);
        configure(cache, s3);
        insert(cache, 8);
        cache.fail_erase_allocation_for_test(point);
        unsigned char output[8];
        check(cache.take(8, output, 8), "Optional spare failure aborted a valid take");
        check(output[0] == 8 && !cache.contains(8), "Take result or ownership incorrect");
        check(cache.stats().pooled_bytes == 0, "Failed optional spare counted unpooled bytes");
        check_accounting(cache);
        insert(cache, 9);
        check_accounting(cache);
    }
}

static void all_leases_are_charged(bool s3) {
    HostExpertCache cache(1000);
    configure(cache, s3);
    std::vector<std::shared_ptr<const uint8_t[]>> leases;
    for (uint64_t key = 0; key < 125; ++key) {
        insert(cache, key);
        leases.push_back(cache.borrow(key, 8));
    }
    check(!insert(cache, 125), "All-leased full cache admitted beyond its physical cap");
    check(cache.stats().resident_bytes == 0 && cache.stats().retired_leased_bytes == 1000,
        "Capacity eviction lost live retired owner accounting");
    for (size_t key = 0; key < leases.size(); ++key)
        check(leases[key][0] == static_cast<unsigned char>(key), "Retired lease data changed");
    check_accounting(cache);
    leases.clear();
    check(insert(cache, 125), "Released leases did not restore admission capacity");
    check_accounting(cache);
}

static void lru_regression() {
    HostExpertCache cache(24);
    insert(cache, 1); insert(cache, 2); insert(cache, 3);
    cache.contains_and_touch(1); // Existing LRU eligibility touch remains unchanged.
    insert(cache, 4);
    check(cache.contains(1) && !cache.contains(2) && cache.contains(3) && cache.contains(4),
        "Default LRU eviction order changed");
    check(std::string(cache.stats().replacement_policy) == "lru", "Default policy changed");
    check(cache.stats().replacement_metadata_bytes == 0, "Default policy allocated S3 metadata");
    check_accounting(cache);
}

static void configuration_and_trim() {
    HostExpertCache cache(1000);
    configure(cache, true);
    expect_throw([&] { cache.set_priority([](uint64_t) { return 1.0; }); }, "S3 accepted competing priority");
    expect_throw([&] { insert(cache, 512); }, "Out-of-range dense key was accepted");
    expect_throw([&] { insert(cache, 1, 9); }, "Larger-than-configured role was accepted");
    for (uint64_t key = 0; key < 250; ++key) {
        insert(cache, key);
        if (key % 5 == 0) { unsigned char output[8]; cache.read(key, output, 8); }
        check_accounting(cache);
    }
    check(cache.stats().s3_ghost_entries != 0, "Capacity misses did not create ghost history");
    cache.trim(0);
    check(cache.stats().resident_bytes == 0, "Trim left managed payloads");
    check_accounting(cache);
    insert(cache, 0);
    insert(cache, 0); // External duplicate replacement does not leave two queue nodes.
    check_accounting(cache);
    HostExpertCache inverse(1000);
    inverse.set_priority([](uint64_t key) { return static_cast<double>(key); });
    expect_throw([&] { configure(inverse, true); }, "Priority cache accepted competing S3");
    HostExpertCache equal_small(80);
    expect_throw([&] { configure(equal_small, true); }, "Equality edge accepted unsupported role size");
}

static void variable_byte_metadata_parity() {
    // This checks the REAL HostExpertCache against the already fixture-tested
    // source-derived metadata algorithm. It is not original-libCacheSim parity.
    HostExpertCache cache(1000);
    configure(cache, true);
    questwend::RamS3FifoMetadata expected(512, 1000);
    uint64_t random = 481;
    for (int request = 0; request < 4000; ++request) {
        random = random * 6364136223846793005ull + 1442695040888963407ull;
        const uint32_t key = static_cast<uint32_t>((random >> 32) % 250);
        const size_t bytes = 4 + key % 5;
        unsigned char output[8];
        const bool expected_hit = expected.resident(key);
        const bool actual_hit = cache.read(key, output, bytes);
        check(actual_hit == expected_hit, "Integrated role-slab hit differs from metadata policy");
        if (expected_hit) {
            expected.reused(key);
            check(output[0] == static_cast<unsigned char>(key), "Integrated payload bytes changed");
        } else {
            const auto intent = expected.prepare_admission(key, bytes);
            while (expected.resident_bytes() > 1000 - bytes)
                expected.commit_capacity_eviction(expected.select_victim());
            check(expected.commit_admission(key, bytes, intent), "Reference declined valid role size");
            check(insert(cache, key, bytes), "Integrated policy rejected eligible unleased slab");
        }
        check(cache.s3_digest_for_test() == expected.debug_digest(),
            "Integrated queue order, frequency or ghost history differs");
        if (request % 127 == 0) check_accounting(cache);
    }
    check_accounting(cache);
}

int main() {
    try {
        successful_reuse_only();
        lru_regression();
        configuration_and_trim();
        variable_byte_metadata_parity();
        for (bool s3 : {false, true}) {
            publication_failures(s3);
            retirement_failure(s3);
            optional_spare_failures(s3);
            all_leases_are_charged(s3);
        }
        std::cout << "HostExpertCache S3FIFO/default-LRU integration checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
