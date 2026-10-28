#pragma once

// Optional expert visualization and telemetry HTTP routes. Registration is
// separate from the inference request path; all captures remain opt-in.
#include "model.h"
#include "runtime.h"
#include "expert_debug.h"
#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

template <class RuntimePtr, class Lock, class ModelPtr>
void register_expert_debug_routes(httplib::Server & srv, RuntimePtr & rt,
                                  Lock & tlock, ModelPtr & model) {
    using json = nlohmann::json;
    using namespace questwend;
    srv.Get("/expert-telemetry", [&](const httplib::Request & req, httplib::Response & res) {
        if (!rt->has_expert_cache()) {
            res.status = 404;
            res.set_content(R"({"status":"unsupported"})", "application/json");
            return;
        }
        int layer = -1;
        try {
            const std::string value = req.get_param_value("layer");
            if (value.empty() || value.size() > 5 ||
                !std::all_of(value.begin(), value.end(), [](unsigned char c) { return c >= '0' && c <= '9'; }))
                throw std::invalid_argument("Invalid layer");
            layer = std::stoi(value);
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content(R"({"error":"layer must be a nonnegative integer"})", "application/json");
            return;
        }
        if (!rt->request_expert_telemetry_layer(layer)) {
            res.status = 400;
            res.set_content(R"({"error":"layer has no expert cache"})", "application/json");
            return;
        }
        if (tlock.try_lock()) {
            rt->refresh_expert_telemetry(layer);
            tlock.unlock();
        }
        const auto view = rt->expert_telemetry_snapshot(layer);
        if (view.layer != layer || view.sequence == 0) {
            res.set_content(json{{"status", "warming"}, {"layer", layer},
                {"layers", model->hparams().n_layer}, {"experts", model->hparams().n_expert}}.dump(),
                "application/json");
            return;
        }
        res.set_content(json{{"status", "ok"}, {"layer", view.layer},
            {"layers", view.layers}, {"experts", view.experts},
            {"sampled_at_unix_ms", view.sampled_at_unix_ms}, {"sequence", view.sequence},
            {"observations_available", view.observations_available},
            {"ram_capacity_bytes", view.ram_capacity_bytes},
            {"ram_resident_bytes", view.ram_resident_bytes},
            {"capture_microseconds", view.capture_microseconds},
            {"capture_total_microseconds", view.capture_total_microseconds},
            {"capture_samples", view.capture_samples},
            {"gpu_mask", view.gpu_mask}, {"ram_mask", view.ram_mask},
            {"observations", view.observations}}.dump(), "application/json");
    });

    // Debug recorder is leased by the operator service. The server itself also
    // enforces the expiry so a crashed browser cannot leave capture running.
    srv.Post("/expert-debug/mode", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const json body = json::parse(req.body);
            const std::string requested = body.at("mode").get<std::string>();
            const int ttl = body.value("ttl_ms", 30000);
            if ((requested != "off" && requested != "overview" && requested != "detailed") ||
                ttl < 0 || ttl > 60000) throw std::invalid_argument("invalid debug mode or TTL");
            const auto mode = requested == "detailed" ? ExpertDebug::Detailed :
                              requested == "overview" ? ExpertDebug::Overview : ExpertDebug::Off;
            ExpertDebug::instance().configure(mode, static_cast<uint64_t>(ttl));
            res.set_content(json{{"mode", requested}, {"ttl_ms", ttl}}.dump(), "application/json");
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content(R"({"error":"invalid debug mode request"})", "application/json");
        }
    });
    srv.Get("/expert-debug/snapshot", [&](const httplib::Request &, httplib::Response & res) {
        const auto mode = ExpertDebug::instance().mode();
        if (mode == ExpertDebug::Off) {
            res.set_content(R"({"status":"off"})", "application/json");
            return;
        }
        if (tlock.try_lock()) {
            rt->refresh_expert_debug();
            tlock.unlock();
        }
        const auto view = ExpertDebug::instance().snapshot();
        res.set_content(json{{"status", view.sequence ? "ok" : "warming"},
            {"mode", mode == ExpertDebug::Detailed ? "detailed" : "overview"},
            {"layers", view.layers}, {"experts", view.experts},
            {"sequence", view.sequence}, {"sampled_at_ms", view.sampled_at_ms},
            {"capture_us", view.capture_us}, {"capture_total_us", view.capture_total_us},
            {"capture_samples", view.capture_samples}, {"dropped_events", view.dropped_events},
            {"phase", view.phase},
            {"gpu", view.gpu}, {"ram", view.ram}, {"observations", view.observations}}
            .dump(), "application/json");
    });
    srv.Get("/expert-debug/events", [&](const httplib::Request & req, httplib::Response & res) {
        uint64_t since = 0;
        try {
            if (req.has_param("since")) since = std::stoull(req.get_param_value("since"));
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content(R"({"error":"invalid event cursor"})", "application/json");
            return;
        }
        uint64_t dropped = 0;
        json rows = json::array();
        for (const auto & event : ExpertDebug::instance().events_since(since, dropped))
            rows.push_back({{"sequence", event.sequence}, {"at_ms", event.at_ms},
                            {"kind", event.kind}, {"layer", event.layer},
                            {"count", event.count}, {"values", event.values}});
        res.set_content(json{{"mode", static_cast<int>(ExpertDebug::instance().mode())},
                             {"dropped_events", dropped}, {"events", rows}}.dump(),
                        "application/json");
    });

}
