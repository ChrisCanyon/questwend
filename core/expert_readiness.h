#pragma once

// One CPU writer per staging span; the owner consumes spans in its original
// order. This object never holds weights, CUDA objects or cache references.
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace questwend {

class ExpertReadiness {
public:
    explicit ExpertReadiness(size_t jobs) : complete_(jobs, false) {}

    void publish(size_t job) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (job >= complete_.size() || complete_[job])
            throw std::logic_error("Expert staging span completed twice or out of bounds");
        complete_[job] = true;
        changed_.notify_all();
    }

    void fail(std::exception_ptr error) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!failure_) failure_ = error;
        changed_.notify_all();
    }

    void wait(size_t job) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (job >= complete_.size()) throw std::out_of_range("Expert staging job index");
        changed_.wait(lock, [&] { return failure_ || complete_[job]; });
        if (failure_) std::rethrow_exception(failure_);
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<bool> complete_;
    std::exception_ptr failure_;
};

} // namespace questwend
