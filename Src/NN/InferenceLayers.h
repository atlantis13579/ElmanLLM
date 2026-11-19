#pragma once

#include "Core/Tensor.h"

namespace elman::inference {

struct Linear {
  Tensor weight, bias;
  Tensor operator()(const Tensor& value) const;
};

struct LayerNorm {
  Tensor weight, bias;
  double epsilon;
  Tensor operator()(const Tensor& value) const;
};

struct Attention {
  Linear query, key, value, output;
  int heads;
  Attention(Linear query, Linear key, Linear value, Linear output, int heads);
  Tensor operator()(const Tensor& hidden, const Tensor& context, bool causal = false) const;
};

struct Conv2d {
  Tensor weight, bias;
  int stride, padding;
  Tensor operator()(const Tensor& value) const;
};

struct GroupNorm {
  Tensor weight, bias;
  int groups;
  double epsilon;
  Tensor operator()(const Tensor& value) const;
};

}  // namespace elman::inference
