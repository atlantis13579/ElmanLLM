#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_silu(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor a = m.float_work(inputs[0]);
  return m.mul(a, [m.g sigmoidWithTensor:a name:nil]);
}
}  // namespace elman::metal
