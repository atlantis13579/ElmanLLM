#include <algorithm>

#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_softmax(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  const int selected = std::max(p.attribute(operator_attr::reduction::kAxis), 0);
  return [m.g softMaxWithTensor:m.float_work(a) axis:selected name:nil];
}
}  // namespace elman::metal
