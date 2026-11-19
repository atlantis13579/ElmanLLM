#pragma once

#include "Core/Tensor.h"

namespace elman {

// Encode one floating-point NCHW RGB image as an 8-bit baseline JPEG.
std::string encode_jpeg(const Tensor& image);

}  // namespace elman
