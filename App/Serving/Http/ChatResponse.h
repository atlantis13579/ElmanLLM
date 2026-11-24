#pragma once

#include "ChatService.h"

namespace elman {
Json chat_usage(size_t prompt, size_t generated);
Json chat_chunk(const ChatRequest& request, const std::string& model, const Json& delta,
                const Json& finish_reason = {}, const Json& usage = {}, bool usage_only = false);
Json chat_completion(const ChatRequest& request, const std::string& model, const GenerationResult& result);
Json chat_error(const std::string& message, const std::string& code, int status);
std::string chat_finish_reason(const GenerationResult& result);
} // namespace elman
