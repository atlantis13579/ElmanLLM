#include "Layers/BatchNorm2d.h"

#include <cmath>
#include <stdexcept>

namespace elman::nn {

BatchNorm2d::BatchNorm2d(int channels, Backend backend, double epsilon, double momentum)
    : channels_(channels), epsilon_(epsilon), momentum_(momentum) {
  if (channels <= 0 || !std::isfinite(epsilon) || epsilon <= 0 ||
      !std::isfinite(momentum) || momentum <= 0 || momentum > 1)
    throw std::invalid_argument("Invalid BatchNorm2d configuration");
  weight_ = register_parameter("weight", Tensor::ones({channels}, backend, true));
  bias_ = register_parameter("bias", Tensor::zeros({channels}, backend, true));
  running_mean_ = register_buffer("running_mean", Tensor::zeros({channels}, backend));
  running_var_ = register_buffer("running_var", Tensor::ones({channels}, backend));
}

Tensor BatchNorm2d::forward(const Tensor& input, uint32_t) {
  if (input.shape().size() != 4 || input.shape()[1] != channels_)
    throw std::invalid_argument("BatchNorm2d expects NCHW input");
  if (!training()) {
    const Shape affine{1, channels_, 1, 1};
    Tensor center = reshape(running_mean_, affine);
    Tensor scale = reshape(rsqrt(add(running_var_, epsilon_)), affine);
    return add(mul(mul(sub(input, center), scale), reshape(weight_, affine)),
               reshape(bias_, affine));
  }
  Tensor result = batch_norm(input, weight_, bias_, epsilon_);
  Tensor rows = reshape(permute(input.detach(), {1, 0, 2, 3}), {channels_, -1});
  const int count = rows.shape()[1];
  if (count < 2) throw std::invalid_argument("BatchNorm2d needs at least two values per channel");
  Tensor average = mean(rows, 1);
  Tensor centered = sub(rows, unsqueeze(average, 1));
  Tensor variance = mean(mul(centered, centered), 1);
  const auto current_mean = average.data(), current_var = variance.data();
  auto updated_mean = running_mean_.data(), updated_var = running_var_.data();
  for (int channel = 0; channel < channels_; ++channel) {
    updated_mean[channel] = static_cast<float>((1 - momentum_) * updated_mean[channel] +
                                               momentum_ * current_mean[channel]);
    updated_var[channel] = static_cast<float>((1 - momentum_) * updated_var[channel] +
                            momentum_ * current_var[channel] * count / (count - 1));
  }
  running_mean_.assign(updated_mean);
  running_var_.assign(updated_var);
  return result;
}

}  // namespace elman::nn
