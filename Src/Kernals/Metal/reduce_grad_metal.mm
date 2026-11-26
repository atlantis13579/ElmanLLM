#include <algorithm>
#include <climits>

#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_reduce_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor input = inputs[0];
  Tensor reduced_value = inputs[1];
  Tensor gradient = inputs[2];
  const int axis = p.attribute(operator_attr::reduction::kAxis);
  NSArray<NSNumber*>* dims = axis < 0 ? axes(0, input.shape.count) : @[ @(axis) ];
  Tensor work = m.float_work(input);
  const int kind = p.attribute(operator_attr::reduction::kGradientKind);
  NSMutableArray<NSNumber*>* kept = [NSMutableArray<NSNumber*> arrayWithArray:input.shape];
  for (NSNumber* dim in dims) {
    kept[dim.unsignedIntegerValue] = @1;
  }
  Tensor reduced = m.reshape(m.float_work(reduced_value), kept);
  Tensor upstream = m.reshape(m.float_work(gradient), kept);
  if (kind == 0) {
    return m.spread(upstream, input.shape);
  }
  if (kind == 1) {
    double count = 1;
    for (NSNumber* dim in dims) {
      count *= input.shape[dim.intValue].intValue;
    }
    return m.spread(m.mul(upstream, m.number(1 / count)), input.shape);
  }
  Tensor equals = m.equal(work, reduced);
  if (kind == 2 || kind == 3) {
    Tensor matches = m.cast(equals, work.dataType);
    return m.mul(m.div(upstream, m.sum(matches, dims)), matches);
  }
  if (kind == 5 || kind == 6) {
    Tensor same_nan = m.both([m.g isNaNWithTensor:work name:nil], [m.g isNaNWithTensor:reduced name:nil]);
    Tensor same = [m.g logicalORWithPrimaryTensor:equals secondaryTensor:same_nan name:nil];
    Shape positions_shape = axis < 0 ? @[ @(p.output_count()) ] : input.shape;
    Tensor index = m.index(positions_shape, std::max(axis, 0));
    if (axis < 0) {
      index = m.reshape(index, input.shape);
    }
    Tensor first = m.min(m.select(same, index, m.integer(INT_MAX)), dims);
    return m.select(m.equal(index, first), upstream, m.number(0, work.dataType));
  }
  if (kind == 4) {
    Tensor zero = m.equal(work, m.number(0, work.dataType));
    Tensor nonzero_product = [m.g reductionProductWithTensor:m.select(zero, m.number(1, work.dataType), work)
                                                      axes:dims
                                                      name:nil];
    Tensor zeros = m.sum(m.cast(zero, MPSDataTypeInt32), dims);
    Tensor zero_result = m.select(m.equal(zeros, m.integer(1)), nonzero_product, m.number(0, work.dataType));
    return m.mul(upstream, m.select(zero, zero_result, m.div(reduced, work)));
  }
  return nil;
}
}  // namespace elman::metal
