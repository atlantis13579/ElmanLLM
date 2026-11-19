#pragma once

#include <filesystem>
#include <functional>

namespace elman {

// The callback writes and closes a temporary file on the destination volume.
void write_file_atomically(const std::filesystem::path& destination,
                           const std::function<void(const std::filesystem::path&)>& write);

}  // namespace elman
