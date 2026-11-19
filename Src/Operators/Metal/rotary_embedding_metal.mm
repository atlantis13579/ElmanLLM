#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_rotary_embedding(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  auto c = inputs[2];
  const int axis = p.output_rank() - 1, half = p.output_shape(axis) / 2;
  auto neg_half = m.neg(m.slice(a, axis, half, half));
  auto prefix = m.slice(a, axis, 0, half);
  auto rotated = [m.g concatTensors:@[ neg_half, prefix ] dimension:axis name:nil];
  return m.add(m.mul(a, b), m.mul(rotated, c));
}
}  // namespace elman::metal
