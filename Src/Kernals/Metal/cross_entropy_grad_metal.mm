#include <cmath>

#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_cross_entropy_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor input = inputs[0];
  Tensor target = inputs[1];
  Tensor gradient = inputs[2];
  const int rank = p.input_rank(KernelInput::A);
  const int classes = p.input_shape(KernelInput::A, rank - 1);
  int rows = 1;
  for (int axis = 0; axis < rank - 1; ++axis) {
    rows *= p.input_shape(KernelInput::A, axis);
  }
  if (!rows) {
    return m.spread(m.number(NAN), shape(p));
  }
  NSArray<NSNumber*>* dims = @[ @(rows), @(classes) ];
  Tensor scores = m.reshape(m.float_work(input), dims);
  Tensor labels = m.reshape(m.cast(target, MPSDataTypeInt32), @[ @(rows), @1 ]);
  Tensor probs = [m.g softMaxWithTensor:scores axis:1 name:nil];
  Tensor positive = m.cast(m.equal(m.index(dims, 1), labels), MPSDataTypeFloat32);
  return m.mul(m.sub(probs, positive), m.mul(m.float_work(gradient), m.number(1. / rows)));
}
}  // namespace elman::metal
