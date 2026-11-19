#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"
#include "Operators/group_norm_params.h"

namespace elman::metal {
Tensor lower_group_norm_input_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  auto input = inputs[0];
  auto weight = inputs[1];
  auto gradient = inputs[2];
  const int batches = p.input_shape(KernelInput::A, 0);
  const int channels = p.input_shape(KernelInput::A, 1);
  const int spatial = p.input_stride(KernelInput::A, 1);
  const int groups = p.attribute(operator_attr::group_norm::kGroups);
  const auto group_shape = @[ @(batches), @(groups), @(channels / groups * spatial) ];
  const auto channel_shape = @[ @(batches), @(channels), @(spatial) ];
  const auto affine_shape = @[ @1, @(channels), @1 ];
  const auto reduce_axes = @[ @2 ];
  auto x = m.reshape(m.float_work(input), group_shape);
  auto center = m.sub(x, m.mean(x, reduce_axes));
  auto eps = m.number(group_norm_epsilon(p));
  auto inverse = m.rsqrt(m.add(m.mean(m.square(center), reduce_axes), eps));
  auto normalized = m.mul(center, inverse);
  weight = m.float_work(weight);
  gradient = m.float_work(gradient);
  auto weighted =
      m.reshape(m.mul(m.reshape(gradient, channel_shape), m.reshape(weight, affine_shape)), group_shape);
  auto correction = m.mul(normalized, m.mean(m.mul(normalized, weighted), @[ @2 ]));
  return m.mul(m.sub(m.sub(weighted, m.mean(weighted, @[ @2 ])), correction), inverse);
}
}  // namespace elman::metal
