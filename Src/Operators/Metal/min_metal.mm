#include <cmath>

#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_min(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  if (!a) {
    return m.filled(p, INFINITY);
  }
  const int axis = p.attribute(operator_attr::reduction::kAxis);
  auto dims = axis < 0 ? axes(0, a.shape.count) : @[ @(axis) ];
  return m.min(m.float_work(a), dims);
}
}  // namespace elman::metal
