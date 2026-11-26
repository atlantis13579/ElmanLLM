#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_argsort(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  return [m.g argSortWithTensor:a axis:axis descending:(p.attribute(1) != 0) name:nil];
}
}  // namespace elman::metal
