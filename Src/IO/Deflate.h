#pragma once

#include "IO/BinaryInput.h"

namespace elman::io {

std::vector<uint8_t> inflate_zlib(Bytes bytes, size_t size);

}  // namespace elman::io
