#pragma once

#include "Layers/Module.h"

namespace elman::nn {

class Dropout final : public Module {
 public:
  explicit Dropout(double probability);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  double probability_;
};

}  // namespace elman::nn
