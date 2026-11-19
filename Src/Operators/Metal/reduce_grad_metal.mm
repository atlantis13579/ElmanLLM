#include <algorithm>
#include <climits>

#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_reduce_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  auto input = inputs[0];
  auto reduced_value = inputs[1];
  auto gradient = inputs[2];
  const int axis = p.attribute(operator_attr::reduction::kAxis);
  auto dims = axis < 0 ? axes(0, input.shape.count) : @[ @(axis) ];
  auto work = m.float_work(input);
  const int kind = p.attribute(operator_attr::reduction::kGradientKind);
  auto* kept = [NSMutableArray<NSNumber*> arrayWithArray:input.shape];
  for (NSNumber* dim in dims) {
    kept[dim.unsignedIntegerValue] = @1;
  }
  auto reduced = m.reshape(m.float_work(reduced_value), kept);
  auto upstream = m.reshape(m.float_work(gradient), kept);
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
  auto equals = m.equal(work, reduced);
  if (kind == 2 || kind == 3) {
    auto matches = m.cast(equals, work.dataType);
    return m.mul(m.div(upstream, m.sum(matches, dims)), matches);
  }
  if (kind == 5 || kind == 6) {
    auto same_nan = m.both([m.g isNaNWithTensor:work name:nil], [m.g isNaNWithTensor:reduced name:nil]);
    auto same = [m.g logicalORWithPrimaryTensor:equals secondaryTensor:same_nan name:nil];
    Shape positions_shape = axis < 0 ? @[ @(p.output_count()) ] : input.shape;
    auto index = m.index(positions_shape, std::max(axis, 0));
    if (axis < 0) {
      index = m.reshape(index, input.shape);
    }
    auto first = m.min(m.select(same, index, m.integer(INT_MAX)), dims);
    return m.select(m.equal(index, first), upstream, m.number(0, work.dataType));
  }
  if (kind == 4) {
    auto zero = m.equal(work, m.number(0, work.dataType));
    auto nonzero_product = [m.g reductionProductWithTensor:m.select(zero, m.number(1, work.dataType), work)
                                                      axes:dims
                                                      name:nil];
    auto zeros = m.sum(m.cast(zero, MPSDataTypeInt32), dims);
    auto zero_result = m.select(m.equal(zeros, m.integer(1)), nonzero_product, m.number(0, work.dataType));
    return m.mul(upstream, m.select(zero, zero_result, m.div(reduced, work)));
  }
  return nil;
}
}  // namespace elman::metal
