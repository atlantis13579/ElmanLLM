#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_relu(Graph m, const Operands& inputs, const KernelParams&) {
  auto a = inputs[0];
  return [m.g reLUWithTensor:m.float_work(a) name:nil];
}
}  // namespace elman::metal
