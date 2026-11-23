#include "Backends/MPSGraph.h"
#include "Core/LocalFunction.h"

#include "Operators/affine_quantization.h"

namespace elman::metal {
Tensor lower_affine_embedding(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor codes = inputs[0];
  Tensor scales = inputs[1];
  Tensor zero_points = inputs[2];
  Tensor indices = inputs[3];
  const int rank = p.input_rank(KernelInput::A);
  const int columns = rank ? p.input_shape(KernelInput::A, rank - 1) : 1;
  int rows = 1;
  for (int axis = 0; axis + 1 < rank; ++axis) {
    rows *= p.input_shape(KernelInput::A, axis);
  }
  if (!columns) {
    return m.zeros(shape(p), MPSDataTypeFloat32);
  }
  Tensor group = m.div(m.index(@[ @(columns) ], 0), m.integer(p.attribute(0)));
  codes = m.reshape(codes, @[ @(rows), @(columns) ]);
  const DType code_type = DType(p.input_dtype(KernelInput::A));
  LocalFunction expand_groups{[&](Tensor parameter) {
    const int axis = parameter.shape.count - 1;
    return parameter.shape[axis].intValue == 1 ? parameter : m.gather(parameter, group, axis);
  }};
  Tensor zero = m.cast(expand_groups(zero_points), MPSDataTypeInt32);
  Tensor min_code = m.integer(quantized_minimum(code_type));
  Tensor max_code = m.integer(quantized_maximum(code_type));
  Tensor valid = m.both([m.g greaterThanOrEqualToWithPrimaryTensor:zero secondaryTensor:min_code name:nil],
                      [m.g lessThanOrEqualToWithPrimaryTensor:zero secondaryTensor:max_code name:nil]);
  Tensor scale = scales;
  if (scale) {
    scale = m.cast(expand_groups(scale), MPSDataTypeFloat32);
    valid = m.both(valid, m.both(m.less(m.number(0), scale), [m.g isFiniteWithTensor:scale name:nil]));
  }
  MPSDataType difference_type = code_type == DType::QInt32 ? MPSDataTypeInt64 : MPSDataTypeInt32;
  Tensor difference = m.sub(m.cast(codes, difference_type), m.cast(zero, difference_type));
  Tensor decoded = m.select(valid, m.mul(m.cast(difference, MPSDataTypeFloat32), scale), m.number(0));
  indices = m.cast(indices, MPSDataTypeInt32);
  Tensor found = m.bounded(indices, rows);
  const int dimensions = int(indices.shape.count);
  Tensor fetched = m.gather(decoded, m.select(found, indices, m.integer(0)), 0);
  return m.select([m.g expandDimsOfTensor:found axis:dimensions name:nil], fetched, m.number(0));
}
}  // namespace elman::metal
