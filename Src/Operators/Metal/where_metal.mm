#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_where(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  Tensor c = inputs[2];
  return m.select(m.truth(m.float_work(a)), m.float_work(b), m.float_work(c));
}
}  // namespace elman::metal
