#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_leaky_relu(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  a = m.float_work(a);
  Tensor slope = m.number(attribute_float(p, operator_attr::activation::kNegativeSlope), a.dataType);
  return m.select(m.less(a, m.number(0, a.dataType)), m.mul(a, slope), a);
}
}  // namespace elman::metal
