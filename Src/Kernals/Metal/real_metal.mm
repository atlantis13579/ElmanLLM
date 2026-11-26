#include "Backends/MPSGraph.h"
#include "Backends/MetalDTypeSupport.h"

namespace elman::metal {
Tensor lower_real(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  if (wide_words(DType(p.output_dtype()))) {
    return m.slice(a, 1, 0, 2);
  }
  return [m.g realPartOfTensor:m.float_work(a) name:nil];
}
}  // namespace elman::metal
