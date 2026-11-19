#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_index_select_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  auto update_shape = a.shape;
  const int rank = p.output_rank();
  auto ids = b.dataType == MPSDataTypeInt64 ? b : m.cast(b, MPSDataTypeInt32);
  auto dims = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  for (int i = 0; i < rank; ++i) {
    [dims addObject:i == axis ? update_shape[i] : @1];
  }
  ids = m.reshape(ids, dims);
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
    return m.filled(p, 0);
  }
  const int n = p.output_count();
  auto flat = @[ @(count) ], extended = @[ @(int64_t(n) + 1) ];
  auto addresses = m.reshape(m.select(ok, destinations, m.integer(n)), flat);
  auto x = m.float_work(a);
  auto sums = [m.g scatterWithUpdatesTensor:m.reshape(x, flat)
                              indicesTensor:addresses
                                      shape:extended
                                       axis:0
                                       mode:MPSGraphScatterModeAdd
                                       name:nil];
  return m.reshape(m.slice(sums, 0, 0, n), shape(p));
}
}  // namespace elman::metal
