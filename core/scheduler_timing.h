#pragma once

#include <chrono>

namespace questwend {

// Timers accumulate owner-thread wall time. Nested measurements overlap and
// must not be added as if they were independent request phases.
class SchedulerTimer {
public:
    explicit SchedulerTimer(double & total) : total_(total), started_(Clock::now()) {}
    ~SchedulerTimer() {
        total_ += std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
    }
private:
    using Clock = std::chrono::steady_clock;
    double & total_;
    Clock::time_point started_;
};

} // namespace questwend
