#include "Optimizer/OptimizerInternal.h"
#include "Core/TensorDispatch.h"
#include <cmath>

namespace elman::optimizer_internal {
DoubleStep double_step(const std::string& algorithm, const Tensor& values,
                       const Tensor& gradients, const Tensor& first_moment,
                       const Tensor& second_moment, const StepParameters& step) {
  auto multiply_add = [&](const Tensor& a, const Tensor& b, const Tensor& c) {
    if (values.backend() == Backend::CUDA)
      return dispatch_tensor("float64_fma", values.shape(), {a, b, c}, {}, DType::Float64);
    return add(mul(a, b), c);
  };
  auto scalar = [&](double value) { return Tensor::from_double_data({}, {value}, values.backend()); };
  auto scaled_add = [&](const Tensor& a, double scale, const Tensor& b) {
    return multiply_add(a, scalar(scale), b);
  };
  auto gradient = mul(gradients, step.gradient_scale);
  if (algorithm != "adamw" && step.weight_decay != 0.)
    gradient = scaled_add(values, step.weight_decay, gradient);
  DoubleStep result;
  if (algorithm == "sgd") {
    result.first_moment = step.momentum == 0. ? gradient : add(mul(first_moment, step.momentum), gradient);
    result.second_moment = second_moment;
    result.value = scaled_add(result.first_moment, -step.rate, values);
    return result;
  }
  // Match lerp's stable branch and Adam's stored intermediate boundaries.
  const auto delta = sub(gradient, first_moment);
  result.first_moment = 1. - step.beta1 < .5 ? scaled_add(delta, 1. - step.beta1, first_moment) :
                                                             scaled_add(delta, -step.beta1, gradient);
  result.second_moment = multiply_add(mul(gradient, 1. - step.beta2), gradient, mul(second_moment, step.beta2));
  const auto denominator = add(div(sqrt(result.second_moment), std::sqrt(step.correction2)), step.epsilon);
  const auto parameter = algorithm == "adamw" && step.weight_decay != 0. ?
      mul(values, 1. - step.rate * step.weight_decay) : values;
  result.value = add(parameter, div(mul(result.first_moment, -step.rate / step.correction1), denominator));
  return result;
}
} // namespace elman::optimizer_internal
