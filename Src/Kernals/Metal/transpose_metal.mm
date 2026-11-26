#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_transpose(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  return m.transpose(a, p.attribute(0), p.attribute(1));
}
}  // namespace elman::metal
