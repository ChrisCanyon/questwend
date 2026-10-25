// Render request fixtures using model metadata only; weights remain on disk.
#include "chat.h"
#include "model.h"
#include "tokenizer.h"
#include "json.hpp"
#include <iostream>

int main(int argc, char ** argv) {
    using namespace questwend;
    using json = nlohmann::json;
    if (argc != 2) return 2;
    auto model = Model::load(argv[1]);
    Tokenizer tokenizer(model->vocab());
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto input = json::parse(line);
            if (input.contains("raw_text")) {
                const auto ids = tokenizer.encode(input["raw_text"], false,
                                                   input.value("parse_special", false));
                std::cout << json{{"ids", ids}, {"text", tokenizer.decode(ids)}}.dump() << '\n';
                continue;
            }
            ChatPromptOptions options;
            options.qwen38_template = true;
            options.reasoning = input.value("enable_thinking", true);
            options.reasoning_effort = input.value("reasoning_effort", std::string("xhigh"));
            options.preserve_thinking = input.value("preserve_thinking", true);
            for (const auto & tool : input.value("tools", json::array())) options.tools_json.push_back(tool.dump());
            std::vector<ChatMessage> messages;
            for (const auto & row : input.at("messages")) {
                ChatMessage message;
                message.role = row.at("role");
                message.content = row.value("content", std::string());
                if (row.contains("reasoning_content")) {
                    message.has_reasoning = true;
                    message.reasoning_content = row.at("reasoning_content");
                }
                for (const auto & call : row.value("tool_calls", json::array())) {
                    const auto & fn = call.at("function");
                    ToolCall tool; tool.name = fn.at("name");
                    for (const auto & item : fn.at("arguments").items())
                        tool.arguments.emplace_back(item.key(), item.value().is_string()
                            ? item.value().get<std::string>() : item.value().dump());
                    message.tool_calls.push_back(std::move(tool));
                }
                messages.push_back(std::move(message));
            }
            const auto prompt = build_qwen_prompt(tokenizer, messages, options);
            std::cout << json{{"text", tokenizer.decode(prompt.ids)}, {"ids", prompt.ids}}.dump() << '\n';
        } catch (const std::exception & error) {
            std::cout << json{{"error", error.what()}}.dump() << '\n';
        }
    }
}
