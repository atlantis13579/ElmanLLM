#include <cmath>

#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"
#include "Operators/group_norm_params.h"

namespace elman::metal {
Tensor lower_group_norm_weight_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor input = inputs[0];
  Tensor gradient = inputs[1];
  if (!input) {
    return m.filled(p, p.input_shape(KernelInput::A, 0) ? NAN : 0);
  }
  const int batches = p.input_shape(KernelInput::A, 0);
  const int channels = p.input_shape(KernelInput::A, 1);
  const int spatial = p.input_stride(KernelInput::A, 1);
  const int groups = p.attribute(operator_attr::group_norm::kGroups);
  NSArray<NSNumber*>* const group_shape = @[ @(batches), @(groups), @(channels / groups * spatial) ];
  NSArray<NSNumber*>* const channel_shape = @[ @(batches), @(channels), @(spatial) ];
  NSArray<NSNumber*>* const reduce_axes = @[ @2 ];
  Tensor x = m.reshape(m.float_work(input), group_shape);
  Tensor center = m.sub(x, m.mean(x, reduce_axes));
  Tensor eps = m.number(group_norm_epsilon(p));
  Tensor inverse = m.rsqrt(m.add(m.mean(m.square(center), reduce_axes), eps));
  Tensor normalized = m.mul(center, inverse);
  gradient = m.float_work(gradient);
  return m.sum(m.mul(m.reshape(gradient, channel_shape), m.reshape(normalized, channel_shape)), @[ @0, @2 ]);
}
}  // namespace elman::metal
