// Host-only parser replay. No model, backend, or tool execution.
#include "qwen_tool_calls.h"
#include <iostream>

int main() {
    using json = nlohmann::json;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto input = json::parse(line);
            questwend::QwenToolSchemas schemas;
            for (const auto & tool : input.at("tools")) {
                const auto & fn = tool.at("function");
                schemas[fn.at("name").get<std::string>()] = fn.value("parameters", json::object());
            }
            const auto result = questwend::parse_qwen_tool_calls(input.at("text"), schemas);
            json calls = json::array();
            for (const auto & call : result.calls)
                calls.push_back({{"name", call.name}, {"arguments", json::parse(call.arguments)}});
            std::cout << json{{"content", result.content}, {"calls", calls},
                {"error", result.error}, {"repaired_headers", result.repaired_headers}}.dump() << '\n';
        } catch (const std::exception & error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
