#pragma once

#include "Layers/Module.h"

namespace elman::nn {

class GroupNorm final : public Module {
 public:
  GroupNorm(int groups, int channels, Backend backend, double epsilon = 1e-5);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  int groups_;
  double epsilon_;
  Tensor weight_, bias_;
};

}  // namespace elman::nn

namespace elman::inference {

struct GroupNorm {
  Tensor weight, bias;
  int groups;
  double epsilon;
  Tensor operator()(const Tensor& value) const;
};

}  // namespace elman::inference
