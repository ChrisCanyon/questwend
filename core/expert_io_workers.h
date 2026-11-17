#pragma once

// Opt-in CPU I/O workers. There are no model, GPU, CUDA or residency operations
// here. The caller retains its existing job partition and upload order.
#include <condition_variable>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace questwend {

struct ExpertIOClosers {
    int (*buffered)(FILE *) = &std::fclose;
    void (*descriptor)(int) = nullptr;
    void (*direct)(void *) = nullptr;
};

// This object lives on one worker's stack until that worker stops. Positioned
// reads on FILE* therefore never share a seek cursor with another worker.
struct ExpertIOFiles {
    std::unordered_map<std::string, FILE *> buffered;
    std::unordered_map<std::string, int> descriptors;
    std::unordered_map<std::string, void *> direct;
    std::vector<uint8_t> aligned;

    explicit ExpertIOFiles(ExpertIOClosers closers = {}) : closers_(closers) {}
    ExpertIOFiles(const ExpertIOFiles &) = delete;
    ExpertIOFiles & operator=(const ExpertIOFiles &) = delete;
    ~ExpertIOFiles() {
        for (auto & entry : buffered) if (entry.second) closers_.buffered(entry.second);
        if (closers_.descriptor)
            for (auto & entry : descriptors) if (entry.second >= 0) closers_.descriptor(entry.second);
        if (closers_.direct)
            for (auto & entry : direct) if (entry.second) closers_.direct(entry.second);
    }

private:
    ExpertIOClosers closers_;
};

// Bounded owner-side instrumentation. No logs, timers or allocations per batch.
struct ExpertIOBatchProfile {
    explicit ExpertIOBatchProfile(bool enabled = false) : enabled(enabled) {}
    void note(int jobs, int active, int configured) {
        if (!enabled) return;
        ++batches;
        total_jobs += static_cast<uint64_t>(jobs);
        active_slots += static_cast<uint64_t>(active);
        broadcast_extra_slots += static_cast<uint64_t>(configured - active);
        ++active_hist[static_cast<size_t>(active)];
        ++jobs_hist[static_cast<size_t>(jobs > 64 ? 65 : jobs)];
    }
    void append_json(FILE * file, bool targeted) const {
        if (!enabled) return;
        std::fprintf(file, ",\"io_batch_profile\":{\"targeted_wake\":%s,\"batches\":%llu,"
            "\"jobs\":%llu,\"active_worker_slots\":%llu,\"broadcast_extra_slots\":%llu,\"active_workers_hist\":[",
            targeted ? "true" : "false", static_cast<unsigned long long>(batches),
            static_cast<unsigned long long>(total_jobs), static_cast<unsigned long long>(active_slots),
            static_cast<unsigned long long>(broadcast_extra_slots));
        for (size_t index = 0; index < active_hist.size(); ++index)
            std::fprintf(file, "%s%llu", index ? "," : "", static_cast<unsigned long long>(active_hist[index]));
        std::fputs("],\"jobs_hist_0_to_64_then_overflow\":[", file);
        for (size_t index = 0; index < jobs_hist.size(); ++index)
            std::fprintf(file, "%s%llu", index ? "," : "", static_cast<unsigned long long>(jobs_hist[index]));
        std::fputs("]}", file);
    }
    bool enabled;
    uint64_t batches = 0, total_jobs = 0, active_slots = 0, broadcast_extra_slots = 0;
    std::array<uint64_t, 65> active_hist{};
    std::array<uint64_t, 66> jobs_hist{};
};

class PersistentExpertIO {
public:
    using Work = std::function<void(int worker_index, ExpertIOFiles & files)>;

    explicit PersistentExpertIO(int workers, ExpertIOClosers closers = {})
        : closers_(closers), targeted_wake_(enabled_flag("QWEN_IO_TARGETED_WAKE")),
          profile_(enabled_flag("QWEN_IO_BATCH_PROFILE")) {
        if (workers < 1 || workers > 64)
            throw std::invalid_argument("Expert I/O worker count must be within 1..64");
        threads_.reserve(static_cast<size_t>(workers));
        if (targeted_wake_) {
            worker_ready_.reserve(static_cast<size_t>(workers));
            for (int index = 0; index < workers; ++index)
                worker_ready_.emplace_back(std::make_unique<std::condition_variable>());
        }
        try {
            for (int index = 0; index < workers; ++index)
                threads_.emplace_back([this, index] { worker_loop(index); });
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    PersistentExpertIO(const PersistentExpertIO &) = delete;
    PersistentExpertIO & operator=(const PersistentExpertIO &) = delete;

    ~PersistentExpertIO() {
        // Normal use is synchronous: every run has finished before return.
        // Serializing here also waits for a submission already holding this
        // mutex before we ask idle workers to close their handles and stop.
        std::lock_guard<std::mutex> serial(submission_mutex_);
        stop_and_join();
    }

    void run(int active_workers, const Work & work, int batch_jobs = 0) {
        if (active_workers == 0) return;
        if (active_workers < 0 || active_workers > static_cast<int>(threads_.size()) || !work || batch_jobs < 0)
            throw std::invalid_argument("Invalid expert I/O batch");
        std::lock_guard<std::mutex> serial(submission_mutex_);
        std::unique_lock<std::mutex> lock(mutex_);
        work_ = &work;
        active_workers_ = active_workers;
        remaining_ = active_workers;
        failure_ = nullptr;
        ++generation_;
        profile_.note(batch_jobs, active_workers, static_cast<int>(threads_.size()));
        notify_workers(active_workers);
        finished_.wait(lock, [this] { return remaining_ == 0; });
        work_ = nullptr;
        const auto failure = failure_;
        lock.unlock();
        if (failure) std::rethrow_exception(failure);
    }

    // Dispatch the same persistent readers, but let their owner service completed
    // CPU staging spans while later reads continue. The callbacks remain alive
    // until ALL workers finish, including when owner progress throws.
    void run_with_owner_progress(int active_workers, const Work & work,
                                 const std::function<void()> & owner_progress, int batch_jobs = 0) {
        if (active_workers < 1 || active_workers > static_cast<int>(threads_.size()) ||
            !work || !owner_progress || batch_jobs < 0)
            throw std::invalid_argument("Invalid pipelined expert I/O batch");
        std::lock_guard<std::mutex> serial(submission_mutex_);
        std::unique_lock<std::mutex> lock(mutex_);
        work_ = &work;
        active_workers_ = active_workers;
        remaining_ = active_workers;
        failure_ = nullptr;
        ++generation_;
        profile_.note(batch_jobs, active_workers, static_cast<int>(threads_.size()));
        notify_workers(active_workers);
        lock.unlock();
        std::exception_ptr owner_failure;
        try { owner_progress(); }
        catch (...) { owner_failure = std::current_exception(); }
        lock.lock();
        finished_.wait(lock, [this] { return remaining_ == 0; });
        work_ = nullptr;
        const auto worker_failure = failure_;
        lock.unlock();
        if (owner_failure) std::rethrow_exception(owner_failure);
        if (worker_failure) std::rethrow_exception(worker_failure);
    }

    // Called only by the existing explicit stats export, never per job.
    void append_batch_stats(FILE * file) {
        std::lock_guard<std::mutex> lock(mutex_);
        profile_.append_json(file, targeted_wake_);
    }

private:
    static bool enabled_flag(const char * name) {
        const char * value = std::getenv(name);
        return value && std::strcmp(value, "1") == 0;
    }
    void notify_workers(int active_workers) {
        if (!targeted_wake_) {
            ready_.notify_all();
            return;
        }
        for (int index = 0; index < active_workers; ++index)
            worker_ready_[static_cast<size_t>(index)]->notify_one();
    }
    void worker_loop(int index) {
        ExpertIOFiles files(closers_);
        uint64_t observed_generation = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            auto & ready = targeted_wake_ ? *worker_ready_[static_cast<size_t>(index)] : ready_;
            ready.wait(lock, [&] { return stopping_ || generation_ != observed_generation; });
            if (stopping_) return;
            observed_generation = generation_;
            if (index >= active_workers_) continue;
            const Work * work = work_;
            lock.unlock();
            std::exception_ptr failure;
            try {
                (*work)(index, files);
            } catch (...) {
                failure = std::current_exception();
            }
            lock.lock();
            if (failure && !failure_) failure_ = failure;
            if (--remaining_ == 0) finished_.notify_one();
        }
    }

    void stop_and_join() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        notify_workers(static_cast<int>(threads_.size()));
        for (auto & thread : threads_) if (thread.joinable()) thread.join();
    }

    ExpertIOClosers closers_;
    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<std::condition_variable>> worker_ready_;
    bool targeted_wake_ = false;
    ExpertIOBatchProfile profile_;
    std::mutex submission_mutex_, mutex_;
    std::condition_variable ready_, finished_;
    const Work * work_ = nullptr;
    uint64_t generation_ = 0;
    int active_workers_ = 0, remaining_ = 0;
    bool stopping_ = false;
    std::exception_ptr failure_;
};

} // namespace questwend
