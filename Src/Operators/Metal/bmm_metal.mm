#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_bmm(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  Tensor product = m.matmul(a, m.cast(b, a.dataType));
  Tensor bias = inputs[2];
  return bias ? m.add(product, m.cast(bias, a.dataType)) : product;
}
}  // namespace elman::metal
