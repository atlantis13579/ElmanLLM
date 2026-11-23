#include "Backends/MPSGraph.h"
#include "Core/LocalFunction.h"

namespace elman::metal {
Tensor lower_categorical(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  Tensor c = inputs[2];
  a = m.float_work(a);
  b = m.float_work(b);
  c = m.float_work(c);
  const int width = p.input_shape(KernelInput::A, 1);
  if (!width) {
    return m.spread(m.integer(-1), shape(p));
  }
  Tensor zero = m.number(0), total = m.slice(b, 1, width - 1, 1);
  Tensor uniform = m.reshape(c, @[ @(p.output_count()), @1 ]);
  Tensor positive = m.less(zero, a);
  Tensor after = m.less(m.mul(uniform, total), b);
  Tensor column = m.index(a.shape, 1);
  Tensor first = m.min(m.select(m.both(positive, after), column, m.integer(width)), @[ @1 ]);
  Tensor last = m.max(m.select(positive, column, m.integer(-1)), @[ @1 ]);
  LocalFunction nonnegative_finite{[&](Tensor t) {
    return m.both([m.g isFiniteWithTensor:t name:nil], [m.g greaterThanOrEqualToWithPrimaryTensor:t
                                                                                  secondaryTensor:zero
                                                                                             name:nil]);
  }};
  Tensor valid_rows = [m.g reductionAndWithTensor:m.both(nonnegative_finite(a), nonnegative_finite(b))
                                           axis:1
                                           name:nil];
  Tensor valid_sample =
      m.both(m.both(nonnegative_finite(uniform), m.less(uniform, m.number(1))), m.less(zero, total));
  return m.select(m.both(valid_rows, valid_sample), m.select(m.less(first, m.integer(width)), first, last),
                  m.integer(-1));
}
}  // namespace elman::metal
