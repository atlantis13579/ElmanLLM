#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_neg(Graph m, const Operands& inputs, const KernelParams&) {
  auto a = inputs[0];
  return m.neg(m.float_work(a));
}
}  // namespace elman::metal
