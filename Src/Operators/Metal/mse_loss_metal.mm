#include <cmath>

#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_mse_loss(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  if (!a) {
    return m.filled(p, NAN);
  }
  return m.mean(m.square(m.sub(m.float_work(a), m.float_work(b))), axes(0, a.shape.count));
}
}  // namespace elman::metal
