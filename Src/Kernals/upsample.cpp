#include "Core/Tensor.h"

#include <limits>
#include <stdexcept>

namespace elman {

Tensor upsample_nearest2d(const Tensor& x, int scale_height, int scale_width) {
  if (x.shape().size() != 4 || scale_height <= 0 || scale_width <= 0)
    throw std::invalid_argument("Nearest upsampling requires NCHW input and positive integer scales");
  const Shape& shape = x.shape();
  if (shape[2] > std::numeric_limits<int>::max() / scale_height ||
      shape[3] > std::numeric_limits<int>::max() / scale_width)
    throw std::overflow_error("Nearest upsampling dimensions overflow");
  Tensor pixels = reshape(x, {shape[0], shape[1], shape[2], 1, shape[3], 1});
  return reshape(repeat(pixels, {1, 1, 1, scale_height, 1, scale_width}),
                 {shape[0], shape[1], shape[2] * scale_height, shape[3] * scale_width});
}

}  // namespace elman
