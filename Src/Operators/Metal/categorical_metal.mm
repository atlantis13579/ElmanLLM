#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_categorical(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  auto c = inputs[2];
  a = m.float_work(a);
  b = m.float_work(b);
  c = m.float_work(c);
  const int width = p.input_shape(KernelInput::A, 1);
  if (!width) {
    return m.spread(m.integer(-1), shape(p));
  }
  auto zero = m.number(0), total = m.slice(b, 1, width - 1, 1);
  auto uniform = m.reshape(c, @[ @(p.output_count()), @1 ]);
  auto positive = m.less(zero, a);
  auto after = m.less(m.mul(uniform, total), b);
  auto column = m.index(a.shape, 1);
  auto first = m.min(m.select(m.both(positive, after), column, m.integer(width)), @[ @1 ]);
  auto last = m.max(m.select(positive, column, m.integer(-1)), @[ @1 ]);
  auto nonnegative_finite = [&](Tensor t) {
    return m.both([m.g isFiniteWithTensor:t name:nil], [m.g greaterThanOrEqualToWithPrimaryTensor:t
                                                                                  secondaryTensor:zero
                                                                                             name:nil]);
  };
  auto valid_rows = [m.g reductionAndWithTensor:m.both(nonnegative_finite(a), nonnegative_finite(b))
                                           axis:1
                                           name:nil];
  auto valid_sample =
      m.both(m.both(nonnegative_finite(uniform), m.less(uniform, m.number(1))), m.less(zero, total));
  return m.select(m.both(valid_rows, valid_sample), m.select(m.less(first, m.integer(width)), first, last),
                  m.integer(-1));
}
}  // namespace elman::metal
