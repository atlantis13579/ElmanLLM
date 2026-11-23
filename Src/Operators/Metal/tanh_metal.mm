#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_tanh(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor a = inputs[0];
  return [m.g tanhWithTensor:m.float_work(a) name:nil];
}
}  // namespace elman::metal
