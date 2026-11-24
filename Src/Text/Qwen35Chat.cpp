#include "Text/HuggingFaceTokenizer.h"
#include <stdexcept>

namespace elman {
namespace {
const char* supported_template =
#include "Text/Qwen35ChatTemplate.inc"
;
std::string trim(const std::string& value) {
    auto text = utf16(value);
    auto space = [](char16_t c) {
        return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
            (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
    };
    size_t begin = 0, end = text.size();
    while (begin < end && space(text[begin])) ++begin;
    while (end > begin && space(text[end - 1])) --end;
    return utf8(text.substr(begin, end - begin));
}
std::string content(const Json& value) {
    if (value.is_null()) return {};
    if (!value.is_array()) return trim(value.string());
    std::string result;
    for (const auto& part : value.array()) {
        if (part.string_or("type", "text") != "text" || !part.contains("text"))
            throw std::invalid_argument("IndexTranslate chat accepts text content only");
        result += part.at("text").string();
    }
    return trim(result);
}
}

std::string HuggingFaceTokenizer::render_qwen35_chat(const Json::Array& messages, bool generation) const {
    if (chat_template_ != supported_template) throw std::invalid_argument("Unsupported Qwen3.5 chat template");
    if (messages.empty()) throw std::invalid_argument("Chat requires at least one message");
    int last_user = -1;
    for (size_t i = 0; i < messages.size(); ++i) if (messages[i].at("role").string() == "user") last_user = int(i);
    if (last_user < 0) throw std::invalid_argument("IndexTranslate chat requires a user message");
    std::string result;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i]; const auto role = message.at("role").string();
        if (role != "system" && role != "user" && role != "assistant") throw std::invalid_argument("Unsupported IndexTranslate chat role");
        if (role == "system" && i != 0) throw std::invalid_argument("System message must be first");
        if (message.contains("tool_calls") && !message.at("tool_calls").is_null() && !message.at("tool_calls").array().empty())
            throw std::invalid_argument("IndexTranslate text chat does not accept tool calls");
        std::string text = content(message.at("content"));
        if (role == "assistant") {
            std::string reasoning = message.string_or("reasoning_content", "");
            if (reasoning.empty()) {
                const auto end = text.rfind("</think>");
                if (end != std::string::npos) {
                    reasoning = text.substr(0, end);
                    const auto begin = reasoning.rfind("<think>");
                    if (begin != std::string::npos) reasoning.erase(0, begin + 7);
                    text.erase(0, end + 8);
                    const auto first = text.find_first_not_of('\n');
                    text = first == std::string::npos ? "" : text.substr(first);
                }
            }
            if (int(i) > last_user) text = "<think>\n" + trim(reasoning) + "\n</think>\n\n" + text;
        }
        result += "<|im_start|>" + role + "\n" + text + "<|im_end|>\n";
    }
    if (generation) result += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return result;
}
} // namespace elman
