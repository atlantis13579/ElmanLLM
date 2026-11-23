#include "Layers/Sequential.h"

#include <utility>

namespace elman::nn {

Module& Sequential::append(std::string name, std::unique_ptr<Module> child) {
  if (order_.size() == order_.capacity()) order_.reserve(order_.size() ? order_.size() * 2 : 1);
  Module& registered = register_module(std::move(name), std::move(child));
  order_.push_back(&registered);
  return registered;
}

Tensor Sequential::forward(const Tensor& input, uint32_t seed) {
  Tensor value = input;
  for (nn::Module* child : order_) {
    value = (*child)(value, seed);
    seed += 0x9e3779b9u;
  }
  return value;
}

}  // namespace elman::nn
