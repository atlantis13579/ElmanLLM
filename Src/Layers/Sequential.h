#pragma once

#include "Layers/Module.h"

namespace elman::nn {

class Sequential final : public Module {
 public:
  Module& append(std::string name, std::unique_ptr<Module> child);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  std::vector<Module*> order_;
};

}  // namespace elman::nn
