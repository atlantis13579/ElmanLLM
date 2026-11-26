#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_concat(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  if (!a || !b) {
    return a ? a : b;
  }
  return [m.g concatTensors:@[ a, b ] dimension:p.attribute(0) name:nil];
}
}  // namespace elman::metal
