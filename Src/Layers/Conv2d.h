#pragma once

#include "Layers/Module.h"

namespace elman::nn {

class Conv2d final : public Module {
 public:
  Conv2d(int inputs, int outputs, int kernel, Backend backend, uint32_t seed = 1,
         int stride = 1, int padding = 0, bool bias = true);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  int stride_, padding_;
  Tensor weight_, bias_;
};

}  // namespace elman::nn

namespace elman::inference {

struct Conv2d {
  Tensor weight, bias;
  int stride, padding;
  Tensor operator()(const Tensor& value) const;
};

}  // namespace elman::inference
