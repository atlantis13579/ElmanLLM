#pragma once

#include "Layers/Module.h"

namespace elman::nn {

class Linear final : public Module {
 public:
  Linear(int inputs, int outputs, Backend backend, uint32_t seed = 1, bool bias = true);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  Tensor weight_, bias_;
};

}  // namespace elman::nn

namespace elman::inference {

struct Linear {
  Tensor weight, bias;
  Tensor operator()(const Tensor& value) const;
};

}  // namespace elman::inference
