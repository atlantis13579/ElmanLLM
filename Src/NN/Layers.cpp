#include "NN/Layers.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace elman::nn {

Module& Sequential::append(std::string name, std::unique_ptr<Module> child) {
  if (order_.size() == order_.capacity()) order_.reserve(order_.size() ? order_.size() * 2 : 1);
  auto& registered = register_module(std::move(name), std::move(child));
  order_.push_back(&registered);
  return registered;
}

Tensor Sequential::forward(const Tensor& input, uint32_t seed) {
  Tensor value = input;
  for (auto* child : order_) {
    value = (*child)(value, seed);
    seed += 0x9e3779b9u;
  }
  return value;
}

Linear::Linear(int inputs, int outputs, Backend backend, uint32_t seed, bool bias) {
  if (inputs <= 0 || outputs <= 0) throw std::invalid_argument("Linear features must be positive");
  const double bound = 1.0 / std::sqrt(inputs);
  weight_ = register_parameter("weight", Tensor::uniform({outputs, inputs}, -bound, bound, seed, backend, true));
  if (bias) bias_ = register_parameter("bias", Tensor::uniform({outputs}, -bound, bound, seed + 1, backend, true));
}

Tensor Linear::forward(const Tensor& input, uint32_t) { return linear(input, weight_, bias_); }

Conv2d::Conv2d(int inputs, int outputs, int kernel, Backend backend, uint32_t seed,
               int stride, int padding, bool bias) : stride_(stride), padding_(padding) {
  if (inputs <= 0 || outputs <= 0 || kernel <= 0 || stride <= 0 || padding < 0)
    throw std::invalid_argument("Invalid Conv2d shape or geometry");
  const double bound = 1.0 / std::sqrt(double(inputs) * kernel * kernel);
  weight_ = register_parameter("weight", Tensor::uniform({outputs, inputs, kernel, kernel}, -bound, bound, seed, backend, true));
  if (bias) bias_ = register_parameter("bias", Tensor::uniform({outputs}, -bound, bound, seed + 1, backend, true));
}

Tensor Conv2d::forward(const Tensor& input, uint32_t) { return conv2d(input, weight_, bias_, stride_, padding_); }

GroupNorm::GroupNorm(int groups, int channels, Backend backend, double epsilon)
    : groups_(groups), epsilon_(epsilon) {
  if (channels <= 0 || groups <= 0 || channels % groups || !std::isfinite(epsilon) || epsilon <= 0)
    throw std::invalid_argument("Invalid GroupNorm configuration");
  weight_ = register_parameter("weight", Tensor::ones({channels}, backend, true));
  bias_ = register_parameter("bias", Tensor::zeros({channels}, backend, true));
}

Tensor GroupNorm::forward(const Tensor& input, uint32_t) { return group_norm(input, groups_, weight_, bias_, epsilon_); }

Dropout::Dropout(double probability) : probability_(probability) {
  if (!std::isfinite(probability) || probability < 0 || probability >= 1)
    throw std::invalid_argument("Dropout probability must be in [0,1)");
}

Tensor Dropout::forward(const Tensor& input, uint32_t seed) { return dropout(input, probability_, seed, training()); }

}  // namespace elman::nn
