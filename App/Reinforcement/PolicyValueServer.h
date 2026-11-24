#pragma once
#include "Core/Backend.h"
#include "IO/Json.h"
#include "Networking/TcpServer.h"

namespace elman {
void serve_policy_value(const Json& config, const std::filesystem::path& base,
                        Backend backend, const std::function<bool()>& stop);
}
