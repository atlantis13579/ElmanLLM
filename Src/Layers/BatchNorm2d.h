#pragma once

#include "Layers/Module.h"

namespace elman::nn {

class BatchNorm2d final : public Module {
 public:
  BatchNorm2d(int channels, Backend backend, double epsilon = 1e-5,
              double momentum = 0.1);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  int channels_;
  double epsilon_, momentum_;
  Tensor weight_, bias_, running_mean_, running_var_;
};

}  // namespace elman::nn
