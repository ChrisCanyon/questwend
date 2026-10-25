#pragma once

#include <chrono>
#include <string>

namespace questwend {

// Partial tool arguments remain buffered; only the existing empty-delta event is sent.
inline bool prepare_stream_event(std::string & event, bool enabled, const std::string & heartbeat,
        std::chrono::steady_clock::time_point now, std::chrono::steady_clock::time_point & last_output) {
    if (event.empty()) {
        if (!enabled || now - last_output < std::chrono::seconds(10)) return false;
        event = heartbeat;
    }
    last_output = now;
    return true;
}

} // namespace questwend
