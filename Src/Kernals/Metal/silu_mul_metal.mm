#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_silu_mul(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor a = m.float_work(inputs[0]);
  Tensor b = inputs[1];
  return m.mul(m.mul(a, [m.g sigmoidWithTensor:a name:nil]), m.float_work(b));
}
}  // namespace elman::metal
