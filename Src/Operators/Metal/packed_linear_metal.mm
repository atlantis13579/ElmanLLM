#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_packed_linear(Graph m, const Operands& inputs, const KernelParams& p) {
  auto input = inputs[0];
  auto weight = inputs[1];
  auto bias = inputs[2];
  const int rank = weight.shape.count;
  weight = m.cast(m.transpose(weight, rank - 2, rank - 1), input.dataType);
  if (input.shape.count == 1) {
    input = m.reshape(input, @[ @1, input.shape[0] ]);
  }
  auto result = m.matmul(input, weight);
  if (bias) {
    result = m.add(result, m.cast(bias, input.dataType));
  }
  const int columns = p.input_shape(KernelInput::B, 0);
  if (!columns) {
    return m.zeros(shape(p), result.dataType);
  }
  const int rows = p.output_count() / columns;
  result = m.reshape(result, @[ @(rows), @(columns) ]);
  auto* partitions = [NSMutableArray<MPSGraphTensor*> array];
  int start = 0;
  for (int group = 0; group < p.attribute(operator_attr::packed_linear::kCount); ++group) {
    const int count = p.attribute(operator_attr::packed_linear::kWidths + group);
    if (count) {
      [partitions addObject:m.reshape(m.slice(result, 1, start, count), @[ @(int64_t(rows) * count) ])];
    }
    start += count;
  }
  return partitions.count == 1 ? partitions.firstObject : [m.g concatTensors:partitions dimension:0 name:nil];
}
}  // namespace elman::metal
