#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "Core/Tensor.h"

namespace elman {

using TensorArchive = std::map<std::string, Tensor>;

std::string model_file_extension(const std::filesystem::path& path);
bool is_tensor_archive(const std::filesystem::path& path);

// NPY provides one tensor under the caller's name; container formats retain
// the tensor names stored in the file.
TensorArchive load_tensor_archive(const std::filesystem::path& path, Backend backend,
                                 const std::string& single_name = "input");

}  // namespace elman
