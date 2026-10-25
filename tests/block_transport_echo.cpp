// No-model control using the same bundled HTTP implementation and payload sizes.
#include "httplib.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static constexpr size_t request_bytes = 51236, response_bytes = 40960;

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Options {
    std::string mode, output, label;
    int port = 0, trials = 20;
    bool disable_expect = false;
};

static Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::runtime_error("Every option needs a value");
        const std::string key = argv[index], value = argv[index + 1];
        if (key == "--mode") options.mode = value;
        else if (key == "--port") options.port = std::stoi(value);
        else if (key == "--trials") options.trials = std::stoi(value);
        else if (key == "--disable-expect") options.disable_expect = std::stoi(value) != 0;
        else if (key == "--output") options.output = value;
        else if (key == "--label") options.label = value;
        else throw std::runtime_error("Unknown option " + key);
    }
    if (options.port < 1024 || options.port > 65535 || options.trials < 1 || options.trials > 40)
        throw std::runtime_error("Invalid bounded port/trial configuration");
    return options;
}

static void serve(const Options & options) {
    httplib::Server server;
    server.set_tcp_nodelay(true);
    server.set_keep_alive_max_count(100000);
    server.set_keep_alive_timeout(300);
    server.set_payload_max_length(4 * 1024 * 1024);
    server.Get("/health", [](const httplib::Request &, httplib::Response & response) {
        response.set_content("{\"ready\":true}", "application/json");
    });
    server.Post("/(0|60)", [](const httplib::Request & request, httplib::Response & response) {
        if (request.body.size() != request_bytes || request.body != std::string(request_bytes, 'q')) {
            response.status = 400;
            response.set_content("Unexpected payload", "text/plain");
            return;
        }
        const int delay_ms = std::stoi(request.path.substr(1));
        const auto started = Clock::now();
        if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        response.set_header("X-Actual-Delay-Ms", std::to_string(elapsed_ms(started)));
        response.set_header("X-Observed-Expect", request.get_header_value("Expect"));
        response.set_content(std::string(response_bytes, 'r'), "application/octet-stream");
    });
    if (!server.listen("127.0.0.1", options.port)) throw std::runtime_error("Loopback bind failed");
}

static Json client(const Options & options) {
    httplib::Client client("127.0.0.1", options.port);
    client.set_tcp_nodelay(true);
    client.set_keep_alive(true);
    client.set_connection_timeout(5);
    client.set_read_timeout(5);
    client.set_write_timeout(5);
    const std::string body(request_bytes, 'q'), expected(response_bytes, 'r');
    Json result{{"label", options.label}, {"disable_expect", options.disable_expect},
        {"bundled_expect_threshold_bytes", CPPHTTPLIB_EXPECT_100_THRESHOLD},
        {"request_bytes", request_bytes}, {"response_bytes", response_bytes},
        {"timer", "std::chrono::steady_clock (Windows QPC)"}, {"cases", Json::array()}};
    for (int delay_ms : {0, 60}) {
        Json rows = Json::array();
        for (int index = 0; index < options.trials + 3; ++index) {
            const auto started = Clock::now();
            const std::string path = "/" + std::to_string(delay_ms);
            auto reply = options.disable_expect
                ? client.Post(path, {{"Expect", ""}}, body, "application/octet-stream")
                : client.Post(path, body, "application/octet-stream");
            const double wall_ms = elapsed_ms(started);
            if (!reply || reply->status != 200 || reply->body != expected)
                throw std::runtime_error("Native echo status/body mismatch");
            const double actual_delay = std::stod(reply->get_header_value("X-Actual-Delay-Ms"));
            if (index >= 3) rows.push_back({{"trial", index - 3}, {"wall_ms", wall_ms},
                {"configured_delay_ms", delay_ms}, {"actual_server_delay_ms", actual_delay},
                {"wall_minus_measured_delay_ms", wall_ms - actual_delay},
                {"server_observed_expect", reply->get_header_value("X-Observed-Expect")}});
        }
        result["cases"].push_back({{"configured_delay_ms", delay_ms}, {"rows", rows}});
    }
    return result;
}

int main(int argc, char ** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.mode == "serve") serve(options);
        else if (options.mode == "client") {
            if (options.output.empty()) throw std::runtime_error("Client needs --output");
            std::ofstream stream(options.output);
            if (!stream) throw std::runtime_error("Cannot open output");
            stream << client(options).dump(2);
        } else throw std::runtime_error("Use --mode serve or client");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "native echo failed: " << error.what() << std::endl;
        return 1;
    }
}
