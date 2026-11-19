#include "NN/InferenceLayers.h"

#include <stdexcept>
#include <utility>

#include "Operators/inference_ops.h"

namespace elman::inference {

Tensor Linear::operator()(const Tensor& value) const { return linear(value, weight, bias); }

Tensor LayerNorm::operator()(const Tensor& value) const { return layer_norm(value, weight, bias, epsilon); }

Attention::Attention(Linear query_, Linear key_, Linear value_, Linear output_, int heads_)
    : query(std::move(query_)), key(std::move(key_)), value(std::move(value_)), output(std::move(output_)), heads(heads_) {
  if (heads <= 0 || query.weight.shape()[0] % heads ||
      query.weight.shape()[0] != key.weight.shape()[0] || query.weight.shape()[0] != value.weight.shape()[0])
    throw std::invalid_argument("Invalid attention head layout");
}

Tensor Attention::operator()(const Tensor& hidden, const Tensor& context, bool causal) const {
  const int batch = hidden.shape().at(0), length = hidden.shape().at(1), width = query.weight.shape()[0];
  const int dimension = width / heads;
  auto split_heads = [&](const Tensor& x) {
    return permute(reshape(x, {batch, x.shape()[1], heads, dimension}), {0, 2, 1, 3});
  };
  auto q = split_heads(query(hidden));
  auto k = split_heads(key(context));
  auto v = split_heads(value(context));
  auto attended = scaled_dot_product_attention(q, k, v, causal);
  return output(reshape(permute(attended, {0, 2, 1, 3}), {batch, length, width}));
}

Tensor Conv2d::operator()(const Tensor& value) const {
  return conv2d_with_channel_bias(value, weight, bias, stride, padding);
}

Tensor GroupNorm::operator()(const Tensor& value) const { return group_norm(value, groups, weight, bias, epsilon); }

}  // namespace elman::inference
