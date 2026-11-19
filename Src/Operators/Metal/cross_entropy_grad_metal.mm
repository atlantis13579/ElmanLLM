#include <cmath>

#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_cross_entropy_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  auto input = inputs[0];
  auto target = inputs[1];
  auto gradient = inputs[2];
  const int rank = p.input_rank(KernelInput::A);
  const int classes = p.input_shape(KernelInput::A, rank - 1);
  int rows = 1;
  for (int axis = 0; axis < rank - 1; ++axis) {
    rows *= p.input_shape(KernelInput::A, axis);
  }
  if (!rows) {
    return m.spread(m.number(NAN), shape(p));
  }
  auto dims = @[ @(rows), @(classes) ];
  auto scores = m.reshape(m.float_work(input), dims);
  auto labels = m.reshape(m.cast(target, MPSDataTypeInt32), @[ @(rows), @1 ]);
  auto probs = [m.g softMaxWithTensor:scores axis:1 name:nil];
  auto positive = m.cast(m.equal(m.index(dims, 1), labels), MPSDataTypeFloat32);
  return m.mul(m.sub(probs, positive), m.mul(m.float_work(gradient), m.number(1. / rows)));
}
}  // namespace elman::metal
