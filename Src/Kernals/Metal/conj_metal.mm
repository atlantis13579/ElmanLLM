#include <climits>

#include "Backends/MPSGraph.h"
#include "Backends/MetalDTypeSupport.h"

namespace elman::metal {
Tensor lower_conj(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  if (wide_words(DType(p.output_dtype()))) {
    return [m.g bitwiseXORWithPrimaryTensor:a
                            secondaryTensor:m.int_vector(@[ @0, @0, @0, @(INT_MIN) ])
                                       name:nil];
  }
  return [m.g conjugateWithTensor:m.float_work(a) name:nil];
}
}  // namespace elman::metal
