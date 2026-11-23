#pragma once

#include "IO/BinaryInput.h"
#include "IO/TensorArchive.h"

namespace elman::io {

Tensor npy_tensor(Bytes bytes, Backend backend);
void save_npy(const std::filesystem::path& path, const Tensor& tensor);
TensorArchive numpy_archive(const std::filesystem::path& file, Backend backend, const std::string& single_name);

}  // namespace elman::io
