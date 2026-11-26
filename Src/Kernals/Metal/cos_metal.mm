#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_cos(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor a = inputs[0];
  return [m.g cosWithTensor:m.float_work(a) name:nil];
}
}  // namespace elman::metal
