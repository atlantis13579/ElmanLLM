#include "Backends/MPSGraph.h"
#include "Backends/MetalDTypeSupport.h"

namespace elman::metal {
Tensor lower_cast(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  const DType dtype = DType(p.output_dtype());
  if (wide_words(dtype) || is_low_precision_float(dtype)) {
    return a;
  }
  return dtype == DType::Bool ? m.truth(m.float_work(a)) : m.cast(a, storage_type(dtype));
}
}  // namespace elman::metal
