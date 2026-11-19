#include <cmath>

#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_cross_entropy(Graph m, const Operands& inputs, const KernelParams& p) {
  auto input = inputs[0];
  auto target = inputs[1];
  if (!input) {
    return m.filled(p, NAN);
  }
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
  auto valid = m.bounded(labels, classes);
  auto shifted = m.sub(scores, m.max(scores, @[ @1 ]));
  auto safe = m.select(valid, labels, m.integer(0));
  auto address = m.add(m.mul(m.index(labels.shape, 0), m.integer(classes)), safe);
  auto correct = m.gather(m.reshape(shifted, @[ @(int64_t(rows) * classes) ]), address, 0);
  auto nll = m.sub(m.log(m.sum(m.exp(shifted), @[ @1 ])), correct);
  return m.mean(m.select(valid, nll, m.number(INFINITY)), @[ @0, @1 ]);
}
}  // namespace elman::metal
