#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_batch_norm(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  Tensor c = inputs[2];
  a = m.float_work(a);
  b = m.float_work(b);
  c = m.float_work(c);
  Shape dims = shape(p, 0);
  NSArray<NSNumber*>* const channel = @[ @1, dims[1], @1, @1 ];
  Tensor mean = m.reshape(b, channel), variance = m.reshape(c, channel);
  return m.mul(m.sub(a, mean), m.rsqrt(m.add(variance, m.number(attribute_float(p, 0)))));
}
}  // namespace elman::metal
