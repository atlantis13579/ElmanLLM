#pragma once

#include "Core/Tensor.h"

namespace elman {

// Encode one floating-point NCHW RGB image as an 8-bit PNG.
std::string encode_png(const Tensor& image);

}  // namespace elman
