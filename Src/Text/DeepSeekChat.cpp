#include "Text/HuggingFaceTokenizer.h"
#include <stdexcept>

namespace elman {
namespace {
const char* supported_template =
#include "Text/DeepSeekChatTemplate.inc"
;
const char* assistant = u8"<\uff5cAssistant\uff5c>";
const char* eos = u8"<\uff5cend\u2581of\u2581sentence\uff5c>";
const char* outputs_end = u8"<\uff5ctool\u2581outputs\u2581end\uff5c>";
}

std::string HuggingFaceTokenizer::render_chat(const Json::Array& messages, bool generation) const {
    if (qwen35_chat_) return render_qwen35_chat(messages, generation);
    if (chat_template_ != supported_template)
        throw std::invalid_argument("Unsupported chat template; expected pinned DeepSeek-R1-Distill-Qwen template");
    if (messages.empty()) throw std::invalid_argument("Chat requires at least one message");
    std::string system;
    for (const Json& message : messages) {
        std::string role = message.at("role").string();
        if (role != "system" && role != "user" && role != "assistant" && role != "tool")
            throw std::invalid_argument("Unsupported chat role: " + role);
        if (role == "system") system = message.at("content").string();
    }
    std::string result = bos_token_ + system;
    bool is_first = false, is_tool = false, is_output_first = true;
    for (const Json& message : messages) {
        std::string role = message.at("role").string();
        const Json& content = message.at("content");
        if (role == "user") {
            is_tool = false;
            result += u8"<\uff5cUser\uff5c>" + content.string();
        } else if (role == "assistant" && content.is_null()) {
            is_tool = false;
            for (const Json& tool : message.at("tool_calls").array()) {
                const Json& function = tool.at("function");
                bool first = !is_first;
                if (first) result += std::string(assistant) + u8"<\uff5ctool\u2581calls\u2581begin\uff5c>";
                else result += "\n";
                result += u8"<\uff5ctool\u2581call\u2581begin\uff5c>" + tool.at("type").string() +
                    u8"<\uff5ctool\u2581sep\uff5c>" + function.at("name").string() + "\n```json\n" +
                    function.at("arguments").string() + "\n```" + u8"<\uff5ctool\u2581call\u2581end\uff5c>";
                // Preserve the pinned template's global first-call state and delimiters.
                if (first) is_first = true;
                else result += std::string(u8"<\uff5ctool\u2581calls\u2581end\uff5c>") + eos;
            }
        } else if (role == "assistant") {
            std::string text = content.string();
            if (is_tool) {
                result += std::string(outputs_end) + text + eos;
                is_tool = false;
            } else {
                size_t end = text.rfind("</think>");
                if (end != std::string::npos) text.erase(0, end + 8);
                result += std::string(assistant) + text + eos;
            }
        } else if (role == "tool") {
            is_tool = true;
            if (is_output_first) {
                result += u8"<\uff5ctool\u2581outputs\u2581begin\uff5c>";
                is_output_first = false;
            } else result += "\n";
            result += u8"<\uff5ctool\u2581output\u2581begin\uff5c>" + content.string() + u8"<\uff5ctool\u2581output\u2581end\uff5c>";
        }
    }
    if (is_tool) result += outputs_end;
    if (generation && !is_tool) result += std::string(assistant) + "<think>\n";
    return result;
}

std::vector<int32_t> HuggingFaceTokenizer::encode_chat(const Json::Array& messages, bool generation) const {
    // The rendered template already includes BOS. Do not apply postprocessing twice.
    return encode(render_chat(messages, generation), false);
}
} // namespace elman
