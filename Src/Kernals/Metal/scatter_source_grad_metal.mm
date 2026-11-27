#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_scatter_source_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  const int axis = p.attribute(kernal_attr::indexing::kAxis);
  const int extent = p.input_shape(KernelInput::A, axis);
  if (!extent) {
    return m.filled(p, 0);
  }
  Tensor ids = b.dataType == MPSDataTypeInt64 ? b : m.cast(b, MPSDataTypeInt32);
  Tensor valid = m.bounded(ids, extent);
  Tensor safe = m.select(valid, m.cast(ids, MPSDataTypeInt32), m.integer(0));
  Shape target = shape(p);
  Tensor address = m.integer(0);
  int count = 1;
  for (int dim = p.input_rank(KernelInput::A) - 1; dim >= 0; --dim) {
    Tensor coordinate = dim == axis ? safe : m.index(target, dim);
    address = m.add(address, m.mul(coordinate, m.integer(p.input_stride(KernelInput::A, dim))));
    count *= p.input_shape(KernelInput::A, dim);
  }
  Tensor result = m.gather(m.reshape(a, @[ @(count) ]), address, 0);
  return m.select(valid, result, m.number(0, a.dataType));
}
}  // namespace elman::metal
