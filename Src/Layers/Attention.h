#pragma once

#include "Layers/Linear.h"

namespace elman::inference {

struct Attention {
  Linear query, key, value, output;
  int heads;
  Attention(Linear query, Linear key, Linear value, Linear output, int heads);
  Tensor operator()(const Tensor& hidden, const Tensor& context, bool causal = false) const;
};

}  // namespace elman::inference
