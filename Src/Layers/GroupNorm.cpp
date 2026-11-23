#include "Layers/GroupNorm.h"
#include "Operators/inference_ops.h"

#include <cmath>
#include <stdexcept>

namespace elman::nn {

GroupNorm::GroupNorm(int groups, int channels, Backend backend, double epsilon)
    : groups_(groups), epsilon_(epsilon) {
  if (channels <= 0 || groups <= 0 || channels % groups || !std::isfinite(epsilon) || epsilon <= 0)
    throw std::invalid_argument("Invalid GroupNorm configuration");
  weight_ = register_parameter("weight", Tensor::ones({channels}, backend, true));
  bias_ = register_parameter("bias", Tensor::zeros({channels}, backend, true));
}

Tensor GroupNorm::forward(const Tensor& input, uint32_t) { return group_norm(input, groups_, weight_, bias_, epsilon_); }

}  // namespace elman::nn

namespace elman::inference {

Tensor GroupNorm::operator()(const Tensor& value) const { return group_norm(value, groups, weight, bias, epsilon); }

}  // namespace elman::inference
