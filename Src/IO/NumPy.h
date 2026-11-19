#pragma once

#include "IO/BinaryInput.h"
#include "IO/TensorArchive.h"

namespace elman::io {

Tensor npy_tensor(Bytes bytes, Backend backend);
TensorArchive numpy_archive(const std::filesystem::path& file, Backend backend, const std::string& single_name);

}  // namespace elman::io
