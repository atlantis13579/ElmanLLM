#pragma once

#include "IO/TensorArchive.h"

namespace elman::io {

TensorArchive torch_weights(const std::filesystem::path& path, Backend backend);

}  // namespace elman::io
