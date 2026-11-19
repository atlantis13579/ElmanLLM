#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"
#include "Operators/matmul_params.h"

namespace elman::metal {
Tensor lower_matmul(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  if (p.attribute(operator_attr::matmul::kTransposeB)) {
    const int rank = b.shape.count;
    b = m.transpose(b, rank - 2, rank - 1);
  }
  auto product = m.matmul(a, m.cast(b, a.dataType));
  auto bias = inputs[2];
  return bias ? m.add(product, m.cast(bias, a.dataType)) : product;
}
}  // namespace elman::metal
