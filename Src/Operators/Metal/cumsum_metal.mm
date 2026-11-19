#include <algorithm>

#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_cumsum(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  const int axis = p.attribute(operator_attr::reduction::kAxis);
  auto work = m.float_work(a);
  if (DType(p.output_dtype()) == DType::Int64) {
    work = m.cast(work, MPSDataTypeInt64);
  }
  return [m.g cumulativeSumWithTensor:work axis:std::max(axis, 0) exclusive:NO reverse:NO name:nil];
}
}  // namespace elman::metal
