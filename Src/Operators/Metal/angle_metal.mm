#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_angle(Graph m, const Operands& inputs, const KernelParams&) {
  auto a = inputs[0];
  a = m.float_work(a);
  return [m.g atan2WithPrimaryTensor:[m.g imaginaryPartOfTensor:a name:nil]
                     secondaryTensor:[m.g realPartOfTensor:a name:nil]
                                name:nil];
}
}  // namespace elman::metal
