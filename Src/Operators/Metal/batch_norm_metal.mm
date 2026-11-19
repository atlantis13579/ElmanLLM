#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_batch_norm(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  auto c = inputs[2];
  a = m.float_work(a);
  b = m.float_work(b);
  c = m.float_work(c);
  auto dims = shape(p, 0);
  const auto channel = @[ @1, dims[1], @1, @1 ];
  auto mean = m.reshape(b, channel), variance = m.reshape(c, channel);
  return m.mul(m.sub(a, mean), m.rsqrt(m.add(variance, m.number(attribute_float(p, 0)))));
}
}  // namespace elman::metal
