// CPU-only worker scheduling fixtures. No model, CUDA or inference dependencies.
#include "expert_io_workers.h"
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>

using questwend::ExpertIOFiles;
using questwend::PersistentExpertIO;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void set_flag(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void changing_active_and_failures() {
    const auto owner = std::this_thread::get_id();
    std::array<std::atomic<int>, 16> observed{};
    std::array<int, 16> expected{};
    std::array<std::thread::id, 16> identities{};
    PersistentExpertIO workers(16);
    for (int repeat = 0; repeat < 12; ++repeat) {
        for (int active : {16, 1, 2, 16, 3, 0, 1, 12, 2, 16}) {
            for (int index = 0; index < active; ++index) ++expected[index];
            workers.run(active, [&](int index, ExpertIOFiles &) {
                const auto identity = std::this_thread::get_id();
                if (identities[index] == std::thread::id{}) identities[index] = identity;
                require(identities[index] == identity, "Worker identity changed");
                ++observed[index];
            }, active * 3);
            for (size_t index = 0; index < observed.size(); ++index)
                require(observed[index] == expected[index], "Active subset ran zero or multiple times");
        }
    }
    for (int active : {1, 3, 16, 2}) {
        std::atomic<int> completed{0};
        bool caught = false;
        try {
            workers.run(active, [&](int index, ExpertIOFiles &) {
                ++completed;
                if (index == 0) throw std::runtime_error("Intentional worker failure");
            }, active);
        } catch (const std::runtime_error &) { caught = true; }
        require(caught && completed == active, "Worker exception escaped before batch completion");
        completed = 0;
        workers.run_with_owner_progress(active, [&](int, ExpertIOFiles &) { ++completed; }, [&] {
            require(std::this_thread::get_id() == owner, "Owner work changed threads");
        }, active);
        require(completed == active, "Owner-progress reuse lost a worker");
        completed = 0;
        caught = false;
        try {
            workers.run_with_owner_progress(active, [&](int, ExpertIOFiles &) { ++completed; }, [&] {
                throw std::runtime_error("Intentional owner failure");
            }, active);
        } catch (const std::runtime_error &) { caught = true; }
        require(caught && completed == active, "Owner failure escaped before worker barrier");
    }
    std::atomic<int> recovered{0};
    workers.run(16, [&](int, ExpertIOFiles &) { ++recovered; }, 16);
    require(recovered == 16, "Pool failed to recover after errors");
    // Idle destructor must wake/join all sixteen, including recently inactive ones.
    workers.run(1, [](int, ExpertIOFiles &) {}, 1);
}

static void histogram() {
    questwend::ExpertIOBatchProfile off;
    off.note(3, 1, 16);
    require(off.batches == 0 && off.jobs_hist[3] == 0, "Disabled histogram mutated");
    questwend::ExpertIOBatchProfile on(true);
    on.note(3, 1, 16);
    on.note(30, 16, 16);
    on.note(1000, 2, 16);
    require(on.batches == 3 && on.total_jobs == 1033 && on.active_slots == 19,
        "Histogram batch totals changed");
    require(on.broadcast_extra_slots == 29 && on.active_hist[1] == 1 &&
        on.active_hist[16] == 1 && on.jobs_hist[65] == 1,
        "Bounded histogram bins changed");
    FILE * file = std::tmpfile();
    require(file != nullptr, "Cannot create synthetic stats stream");
    on.append_json(file, true);
    std::rewind(file);
    std::array<char, 4096> text{};
    const auto bytes = std::fread(text.data(), 1, text.size() - 1, file);
    std::fclose(file);
    require(bytes > 0 && std::strstr(text.data(), "\"targeted_wake\":true") &&
        std::strstr(text.data(), "\"batches\":3"), "Stats JSON fields missing");
}

int main() {
    try {
        histogram();
        set_flag("QWEN_IO_BATCH_PROFILE", "1");
        for (const char * mode : {"0", "1"}) {
            set_flag("QWEN_IO_TARGETED_WAKE", mode);
            changing_active_and_failures();
            for (int repeat = 0; repeat < 8; ++repeat) {
                PersistentExpertIO idle(16);
                idle.run(repeat % 3, [](int, ExpertIOFiles &) {}, repeat % 3);
            }
        }
        std::cout << "Targeted/broadcast wake, changing subsets, worker/owner failure, reuse, idle destruction and bounded histogram passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Targeted worker test failed: " << error.what() << '\n';
        return 1;
    }
}
