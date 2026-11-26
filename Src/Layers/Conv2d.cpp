#include "Layers/Conv2d.h"
#include "Kernals/inference_ops.h"

#include <cmath>
#include <stdexcept>

namespace elman::nn {

Conv2d::Conv2d(int inputs, int outputs, int kernel, Backend backend, uint32_t seed,
               int stride, int padding, bool bias) : stride_(stride), padding_(padding) {
  if (inputs <= 0 || outputs <= 0 || kernel <= 0 || stride <= 0 || padding < 0)
    throw std::invalid_argument("Invalid Conv2d shape or geometry");
  const double bound = 1.0 / std::sqrt(double(inputs) * kernel * kernel);
  weight_ = register_parameter("weight", Tensor::uniform({outputs, inputs, kernel, kernel}, -bound, bound, seed, backend, true));
  if (bias) bias_ = register_parameter("bias", Tensor::uniform({outputs}, -bound, bound, seed + 1, backend, true));
}

Tensor Conv2d::forward(const Tensor& input, uint32_t) { return conv2d(input, weight_, bias_, stride_, padding_); }

}  // namespace elman::nn

namespace elman::inference {

Tensor Conv2d::operator()(const Tensor& value) const {
  return conv2d_with_channel_bias(value, weight, bias, stride, padding);
}

}  // namespace elman::inference
