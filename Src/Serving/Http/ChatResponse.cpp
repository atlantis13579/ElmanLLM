#include "ChatResponse.h"

namespace elman {
namespace {
Json ids_json(const std::vector<int32_t>& ids) {
    Json::Array result;
    for (int32_t id : ids) result.emplace_back(id);
    return result;
}
} // namespace

Json chat_usage(size_t prompt, size_t generated) {
    return Json::Object{{"prompt_tokens", double(prompt)}, {"completion_tokens", double(generated)}, {"total_tokens", double(prompt + generated)}};
}
Json chat_chunk(const ChatRequest& request, const std::string& model, const Json& delta,
                const Json& finish_reason, const Json& usage, bool usage_only) {
    Json::Array choices;
    if (!usage_only) choices.emplace_back(Json::Object{{"index", 0}, {"delta", delta}, {"finish_reason", finish_reason}, {"logprobs", Json{}}});
    Json::Object result{{"id", request.id}, {"object", "chat.completion.chunk"}, {"created", request.created}, {"model", model}, {"choices", choices}};
    if (request.include_usage) result.emplace("usage", usage);
    return result;
}
std::string chat_finish_reason(const GenerationResult& result) {
    if (result.finish_reason == "length") return "length";
    if (result.finish_reason == "eos" || result.finish_reason == "stop") return "stop";
    throw ChatError(503, "generation_interrupted", "Generation was interrupted");
}
Json chat_completion(const ChatRequest& request, const std::string& model, const GenerationResult& result) {
    Json::Object message{{"role", "assistant"}, {"content", result.text}};
    if (request.token_ids) message.emplace("token_ids", ids_json(result.ids));
    Json choice = Json::Object{{"index", 0}, {"message", std::move(message)},
        {"finish_reason", chat_finish_reason(result)}, {"logprobs", Json{}}};
    return Json::Object{{"id", request.id}, {"object", "chat.completion"}, {"created", request.created}, {"model", model},
        {"choices", Json::Array{choice}}, {"usage", chat_usage(request.prompt_tokens, result.ids.size())}};
}
Json chat_error(const std::string& message, const std::string& code, int status) {
    return Json::Object{{"error", Json::Object{{"message", message}, {"code", code}, {"param", Json{}},
        {"type", status < 500 ? "invalid_request_error" : "server_error"}}}};
}
} // namespace elman
