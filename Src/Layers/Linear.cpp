#include "Layers/Linear.h"
#include "Operators/inference_ops.h"

#include <cmath>
#include <stdexcept>

namespace elman::nn {

Linear::Linear(int inputs, int outputs, Backend backend, uint32_t seed, bool bias) {
  if (inputs <= 0 || outputs <= 0) throw std::invalid_argument("Linear features must be positive");
  const double bound = 1.0 / std::sqrt(inputs);
  weight_ = register_parameter("weight", Tensor::uniform({outputs, inputs}, -bound, bound, seed, backend, true));
  if (bias) bias_ = register_parameter("bias", Tensor::uniform({outputs}, -bound, bound, seed + 1, backend, true));
}

Tensor Linear::forward(const Tensor& input, uint32_t) { return linear(input, weight_, bias_); }

}  // namespace elman::nn

namespace elman::inference {

Tensor Linear::operator()(const Tensor& value) const { return linear(value, weight, bias); }

}  // namespace elman::inference
