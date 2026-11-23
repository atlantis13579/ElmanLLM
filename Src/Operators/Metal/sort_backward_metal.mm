#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_sort_backward(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  Shape update_shape = a.shape;
  const int rank = p.output_rank();
  Tensor ids = b.dataType == MPSDataTypeInt64 ? b : m.cast(b, MPSDataTypeInt32);
  Tensor ok = m.bounded(ids, p.output_shape(axis));
  ids = m.select(ok, m.cast(ids, MPSDataTypeInt32), m.integer(0));
  Tensor destinations = m.integer(0);
  int stride = 1, count = 1;
  for (int dim = rank - 1; dim >= 0; --dim) {
    Tensor coordinate = dim == axis ? ids : m.index(update_shape, dim);
    destinations = m.add(destinations, m.mul(m.integer(stride), coordinate));
    stride *= p.output_shape(dim);
    count *= update_shape[dim].intValue;
  }
  if (!count) {
    return m.filled(p, 0);
  }
  const int n = p.output_count();
  NSArray<NSNumber*>* flat = @[ @(count) ], *extended = @[ @(int64_t(n) + 1) ];
  Tensor addresses = m.reshape(m.select(ok, destinations, m.integer(n)), flat);
  Shape result_shape = shape(p);
  // Scatter the greatest source ordinal, then gather that value for each
  // destination. A repeated index has a stable last-writer result.
  Tensor ordinal = m.add(m.index(flat, 0), m.integer(1));
  Tensor winners = [m.g scatterWithUpdatesTensor:ordinal
                                 indicesTensor:addresses
                                         shape:extended
                                          axis:0
                                          mode:MPSGraphScatterModeMax
                                          name:nil];
  winners = m.slice(winners, 0, 0, n);
  Tensor indices = m.maximum(m.sub(winners, m.integer(1)), m.integer(0));
  Tensor latest = m.reshape(m.gather(m.reshape(a, flat), indices, 0), result_shape);
  Tensor written = m.reshape(m.truth(winners), result_shape);
  return m.select(written, latest, m.number(0, a.dataType));
}
}  // namespace elman::metal
