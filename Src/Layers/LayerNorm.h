#pragma once

#include "Core/Tensor.h"

namespace elman::inference {

struct LayerNorm {
  Tensor weight, bias;
  double epsilon;
  Tensor operator()(const Tensor& value) const;
};

}  // namespace elman::inference
