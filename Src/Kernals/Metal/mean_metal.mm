#include <cmath>

#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_mean(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  if (!a) {
    return m.filled(p, NAN);
  }
  const int axis = p.attribute(kernal_attr::reduction::kAxis);
  NSArray<NSNumber*>* dims = axis < 0 ? axes(0, a.shape.count) : @[ @(axis) ];
  return m.mean(m.float_work(a), dims);
}
}  // namespace elman::metal
