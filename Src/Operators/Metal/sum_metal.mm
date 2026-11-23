#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_sum(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  if (!a) {
    return m.filled(p, 0);
  }
  const int axis = p.attribute(operator_attr::reduction::kAxis);
  NSArray<NSNumber*>* dims = axis < 0 ? axes(0, a.shape.count) : @[ @(axis) ];
  Tensor work = m.float_work(a);
  const DType dtype = DType(p.output_dtype());
  if (is_standard_integer(dtype) && dtype != DType::Bool) {
    work = m.cast(work, storage_type(dtype));
  }
  return m.sum(work, dims);
}
}  // namespace elman::metal
