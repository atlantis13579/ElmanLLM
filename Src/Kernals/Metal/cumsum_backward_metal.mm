#include <algorithm>

#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_cumsum_backward(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  const int axis = p.attribute(kernal_attr::reduction::kAxis);
  Tensor work = m.float_work(a);
  if (DType(p.output_dtype()) == DType::Int64) {
    work = m.cast(work, MPSDataTypeInt64);
  }
  return [m.g cumulativeSumWithTensor:work axis:std::max(axis, 0) exclusive:NO reverse:YES name:nil];
}
}  // namespace elman::metal
