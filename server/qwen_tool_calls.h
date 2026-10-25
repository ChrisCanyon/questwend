#pragma once

#include "json.hpp"
#include <map>
#include <string>
#include <vector>

namespace questwend {

struct QwenToolCall { std::string name; std::string arguments; };
struct QwenToolParse {
    std::string content;
    std::vector<QwenToolCall> calls;
    std::string error;
    int repaired_headers = 0;
};
using QwenToolSchemas = std::map<std::string, nlohmann::json>;

inline void tool_skip_space(const std::string & text, size_t & at) {
    while (at < text.size() && (text[at] == ' ' || text[at] == '\n' ||
           text[at] == '\r' || text[at] == '\t')) ++at;
}

inline bool tool_take(const std::string & text, size_t & at, const std::string & token) {
    if (text.compare(at, token.size(), token) != 0) return false;
    at += token.size();
    return true;
}

inline bool tool_type_matches(const nlohmann::json & value, const nlohmann::json & schema) {
    if (!schema.is_object() || !schema.contains("type")) return true;
    const auto & type = schema["type"];
    if (type.is_array()) {
        for (const auto & item : type)
            if (tool_type_matches(value, {{"type", item}})) return true;
        return false;
    }
    if (!type.is_string()) return false;
    const auto name = type.get<std::string>();
    return name == "string" ? value.is_string() : name == "object" ? value.is_object() :
           name == "array" ? value.is_array() : name == "integer" ? value.is_number_integer() :
           name == "number" ? value.is_number() : name == "boolean" ? value.is_boolean() :
           name == "null" ? value.is_null() : false;
}

// Check only the envelope, required arguments and top-level types here. The
// consumer's tool executor remains responsible for full schema validation.
inline std::string tool_arguments_error(const nlohmann::json & args,
                                        const nlohmann::json & schema) {
    if (!args.is_object()) return "Tool arguments must be an object.";
    if (schema.contains("required") && schema["required"].is_array())
        for (const auto & key : schema["required"])
            if (key.is_string() && !args.contains(key.get<std::string>()))
                return "Tool call is missing a required argument.";
    const auto props = schema.value("properties", nlohmann::json::object());
    for (auto it = args.begin(); it != args.end(); ++it) {
        if (!props.contains(it.key())) return "Tool call contains an undeclared argument.";
        if (!tool_type_matches(it.value(), props[it.key()])) return "Tool argument has the wrong type.";
    }
    return {};
}

// Parse sequentially, including entire parameter values, so code containing
// XML-looking strings cannot accidentally become a second tool call. Never
// dispatch a partial batch: one invalid block invalidates all calls in it.
inline QwenToolParse parse_qwen_tool_calls(const std::string & text,
                                          const QwenToolSchemas & schemas) {
    using json = nlohmann::json;
    QwenToolParse out;
    if (schemas.empty()) { out.content = text; return out; }
    auto fail = [&](const std::string & error) {
        out.error = error;
        out.calls.clear();
        return out;
    };
    size_t at = 0;
    // The caller removes the reasoning channel first. Do not search for
    // </think> here: that string may legitimately occur in a file argument.
    while (at < text.size()) {
        const size_t begin = text.find("<tool_call>", at);
        if (begin == std::string::npos) { out.content += text.substr(at); break; }
        out.content += text.substr(at, begin - at);
        at = begin + 11;
        tool_skip_space(text, at);
        const bool canonical = tool_take(text, at, "<function=");
        const bool repair = !canonical && tool_take(text, at, "<parameter=");
        if (!canonical && !repair) return fail("Expected a function header inside tool_call.");
        const size_t end_name = text.find('>', at);
        if (end_name == std::string::npos) return fail("Incomplete tool function header.");
        const std::string name = text.substr(at, end_name - at);
        const auto found = schemas.find(name);
        if (found == schemas.end()) return fail("Tool call names an undeclared function.");
        const json & schema = found->second;
        const auto props = schema.value("properties", json::object());
        json args = json::object();
        at = end_name + 1;
        tool_skip_space(text, at);
        while (tool_take(text, at, "<parameter=")) {
            const size_t end_key = text.find('>', at);
            if (end_key == std::string::npos) return fail("Incomplete tool parameter header.");
            const std::string key = text.substr(at, end_key - at);
            if (args.contains(key)) return fail("Duplicate tool argument.");
            if (!props.contains(key)) return fail("Tool call contains an undeclared argument.");
            at = end_key + 1;
            const size_t end_value = text.find("</parameter>", at);
            if (end_value == std::string::npos) return fail("Incomplete tool parameter value.");
            std::string value = text.substr(at, end_value - at);
            if (!value.empty() && value.front() == '\n') value.erase(0, 1);
            if (!value.empty() && value.back() == '\n') value.pop_back();
            // Preserve file contents such as "123", "true" and JSON exactly.
            if (props[key].value("type", json()) == "string") args[key] = value;
            else {
                try { args[key] = json::parse(value); }
                catch (...) { args[key] = value; }
            }
            at = end_value + 12;
            tool_skip_space(text, at);
        }
        // Header repair is deliberately narrow: only the observed wrong tag,
        // a declared function, valid arguments, and both correct closing tags.
        if (!tool_take(text, at, "</function>")) return fail("Missing tool function closing tag.");
        tool_skip_space(text, at);
        if (!tool_take(text, at, "</tool_call>")) return fail("Missing tool_call closing tag.");
        const auto error = tool_arguments_error(args, schema);
        if (!error.empty()) return fail(error);
        out.calls.push_back({name, args.dump()});
        if (repair) ++out.repaired_headers;
    }
    while (!out.content.empty() && (out.content.back() == '\n' || out.content.back() == ' '))
        out.content.pop_back();
    return out;
}

} // namespace questwend
