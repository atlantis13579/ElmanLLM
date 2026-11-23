#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_rsqrt(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor a = inputs[0];
  return [m.g reciprocalSquareRootWithTensor:m.float_work(a) name:nil];
}
}  // namespace elman::metal
