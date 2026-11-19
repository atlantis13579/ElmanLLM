#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_logical_and(Graph m, const Operands& inputs, const KernelParams&) {
  auto a = inputs[0];
  auto b = inputs[1];
  return m.both(m.truth(m.float_work(a)), m.truth(m.float_work(b)));
}
}  // namespace elman::metal
