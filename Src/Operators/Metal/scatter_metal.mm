#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_scatter(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  auto c = inputs[2];
  if (!c) {
    return a;
  }
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  auto update_shape = c.shape;
  const int rank = p.output_rank();
  auto ids = b.dataType == MPSDataTypeInt64 ? b : m.cast(b, MPSDataTypeInt32);
  auto ok = m.bounded(ids, p.output_shape(axis));
  ids = m.select(ok, m.cast(ids, MPSDataTypeInt32), m.integer(0));
  auto destinations = m.integer(0);
  int stride = 1, count = 1;
  for (int dim = rank - 1; dim >= 0; --dim) {
    auto coordinate = dim == axis ? ids : m.index(update_shape, dim);
    destinations = m.add(destinations, m.mul(m.integer(stride), coordinate));
    stride *= p.output_shape(dim);
    count *= update_shape[dim].intValue;
  }
  if (!count) {
    return a;
  }
  const int n = p.output_count();
  auto flat = @[ @(count) ], extended = @[ @(int64_t(n) + 1) ];
  auto addresses = m.reshape(m.select(ok, destinations, m.integer(n)), flat);
  auto result_shape = shape(p);
  // Scatter the greatest source ordinal, then gather that value for each
  // destination. A repeated index has a stable last-writer result.
  auto ordinal = m.add(m.index(flat, 0), m.integer(1));
  auto winners = [m.g scatterWithUpdatesTensor:ordinal
                                 indicesTensor:addresses
                                         shape:extended
                                          axis:0
                                          mode:MPSGraphScatterModeMax
                                          name:nil];
  winners = m.slice(winners, 0, 0, n);
  auto indices = m.maximum(m.sub(winners, m.integer(1)), m.integer(0));
  auto latest = m.reshape(m.gather(m.reshape(c, flat), indices, 0), result_shape);
  auto written = m.reshape(m.truth(winners), result_shape);
  return m.select(written, latest, a);
}
}  // namespace elman::metal
