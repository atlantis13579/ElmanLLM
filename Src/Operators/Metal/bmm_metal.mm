#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_bmm(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  auto product = m.matmul(a, m.cast(b, a.dataType));
  auto bias = inputs[2];
  return bias ? m.add(product, m.cast(bias, a.dataType)) : product;
}
}  // namespace elman::metal
