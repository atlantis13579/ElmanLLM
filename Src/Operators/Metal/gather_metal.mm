#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_gather(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  const int extent = p.input_shape(KernelInput::A, axis);
  if (!extent) {
    return m.filled(p, 0);
  }
  auto ids = b.dataType == MPSDataTypeInt64 ? b : m.cast(b, MPSDataTypeInt32);
  auto valid = m.bounded(ids, extent);
  auto safe = m.select(valid, m.cast(ids, MPSDataTypeInt32), m.integer(0));
  auto target = shape(p);
  auto address = m.integer(0);
  int count = 1;
  for (int dim = p.input_rank(KernelInput::A) - 1; dim >= 0; --dim) {
    auto coordinate = dim == axis ? safe : m.index(target, dim);
    address = m.add(address, m.mul(coordinate, m.integer(p.input_stride(KernelInput::A, dim))));
    count *= p.input_shape(KernelInput::A, dim);
  }
  auto result = m.gather(m.reshape(a, @[ @(count) ]), address, 0);
  return m.select(valid, result, m.number(0, a.dataType));
}
}  // namespace elman::metal
