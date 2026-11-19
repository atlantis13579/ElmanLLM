#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>

#include "IO/Json.h"

namespace elman::tools {

std::string sha256_file(const std::filesystem::path& path);
std::filesystem::path executable_path();
void write_json(std::ostream& stream, const Json& value);
void write_json_atomically(const std::filesystem::path& path, const Json& value);

}  // namespace elman::tools
