#include "stream_heartbeat.h"
#include <iostream>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    using namespace std::chrono;
    using questwend::prepare_stream_event;
    try {
        const steady_clock::time_point start{};
        auto last = start;
        std::string event;
        const std::string heartbeat = "data: {\"choices\":[{\"delta\":{}}]}\n\n";
        require(!prepare_stream_event(event, true, heartbeat, start + seconds(9), last),
                "No keepalive before the interval");
        require(event.empty() && last == start, "Buffered content must remain untouched");
        require(prepare_stream_event(event, true, heartbeat, start + seconds(10), last),
                "Keepalive must be emitted at the interval");
        require(event == heartbeat && last == start + seconds(10), "Use the existing empty delta");
        event.clear();
        require(!prepare_stream_event(event, true, heartbeat, start + seconds(19), last),
                "Keepalives must be throttled");
        require(!prepare_stream_event(event, false, heartbeat, start + seconds(100), last),
                "Disabled heartbeat must not emit empty events");
        event = "actual text";
        require(prepare_stream_event(event, false, heartbeat, start + seconds(100), last),
                "Disabled heartbeat must still send real output");
        require(event == "actual text" && last == start + seconds(100), "Real output resets the interval");
        event.clear();
        require(!prepare_stream_event(event, true, heartbeat, start + seconds(109), last),
                "Real output postpones the next heartbeat");
        require(prepare_stream_event(event, true, heartbeat, start + seconds(110), last),
                "Heartbeat resumes after real output goes silent");
        std::cout << "10 stream heartbeat checks passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
