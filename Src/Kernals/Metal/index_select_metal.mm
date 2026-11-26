#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_index_select(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  const int extent = p.input_shape(KernelInput::A, axis);
  if (!extent) {
    return m.filled(p, 0);
  }
  Tensor ids = b.dataType == MPSDataTypeInt64 ? b : m.cast(b, MPSDataTypeInt32);
  Tensor valid = m.bounded(ids, extent);
  Tensor safe = m.select(valid, m.cast(ids, MPSDataTypeInt32), m.integer(0));
  NSMutableArray<NSNumber*>* dims = [NSMutableArray<NSNumber*> array];
  for (int i = 0; i < p.output_rank(); ++i) {
    [dims addObject:@(i == axis ? p.output_shape(i) : 1)];
  }
  return m.select(m.reshape(valid, dims), m.gather(a, safe, axis), m.number(0, a.dataType));
}
}  // namespace elman::metal
