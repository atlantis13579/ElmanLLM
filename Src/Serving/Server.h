#pragma once

#include <filesystem>

namespace elman {

void run_server(const std::filesystem::path& config_path, bool open_web);

} // namespace elman
