#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_expand(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  return m.spread(a, shape(p));
}
}  // namespace elman::metal
