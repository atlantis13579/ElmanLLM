#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_squeeze(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  return m.reshape(a, shape(p));
}
}  // namespace elman::metal
