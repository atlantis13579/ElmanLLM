#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_logical_not(Graph m, const Operands& inputs, const KernelParams&) {
  auto a = inputs[0];
  return [m.g notWithTensor:m.truth(m.float_work(a)) name:nil];
}
}  // namespace elman::metal
