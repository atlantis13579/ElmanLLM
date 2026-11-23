#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_layer_norm(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor input = inputs[0];
  Tensor weight = inputs[1];
  Tensor bias = inputs[2];
  input = m.float_work(input);
  weight = m.float_work(weight);
  bias = m.float_work(bias);
  Shape dims = shape(p, 0);
  const int rank = dims.count, suffix = p.attribute(operator_attr::normalization::kTrailingDimensions);
  NSArray<NSNumber*>* reduce_axes = axes(rank - suffix, rank);
  Tensor center = m.sub(input, m.mean(input, reduce_axes));
  Tensor epsilon = m.number(attribute_float(p, operator_attr::normalization::kEpsilon));
  Tensor result = m.mul(center, m.rsqrt(m.add(m.mean(m.square(center), reduce_axes), epsilon)));
  if (weight) {
    result = m.mul(result, weight);
  }
  return bias ? m.add(result, bias) : result;
}
}  // namespace elman::metal
