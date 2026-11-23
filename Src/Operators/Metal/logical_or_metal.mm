#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_logical_or(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  return [m.g logicalORWithPrimaryTensor:m.truth(m.float_work(a))
                         secondaryTensor:m.truth(m.float_work(b))
                                    name:nil];
}
}  // namespace elman::metal
