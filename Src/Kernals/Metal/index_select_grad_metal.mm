#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_index_select_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  Shape update_shape = a.shape;
  const int rank = p.output_rank();
  Tensor ids = b.dataType == MPSDataTypeInt64 ? b : m.cast(b, MPSDataTypeInt32);
  NSMutableArray<NSNumber*>* dims = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  for (int i = 0; i < rank; ++i) {
    [dims addObject:i == axis ? update_shape[i] : @1];
  }
  ids = m.reshape(ids, dims);
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
  Tensor x = m.float_work(a);
  Tensor sums = [m.g scatterWithUpdatesTensor:m.reshape(x, flat)
                              indicesTensor:addresses
                                      shape:extended
                                       axis:0
                                       mode:MPSGraphScatterModeAdd
                                       name:nil];
  return m.reshape(m.slice(sums, 0, 0, n), shape(p));
}
}  // namespace elman::metal
