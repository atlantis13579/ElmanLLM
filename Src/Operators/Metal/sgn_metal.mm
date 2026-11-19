#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_sgn(Graph m, const Operands& inputs, const KernelParams&) {
  auto a = inputs[0];
  a = m.float_work(a);
  auto* magnitude = [m.g absoluteWithTensor:a name:nil];
  auto* normalized = m.div(a, m.cast(magnitude, a.dataType));
  return m.select(m.equal(magnitude, m.number(0, magnitude.dataType)), m.number(0, a.dataType), normalized);
}
}  // namespace elman::metal
