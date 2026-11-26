#include "Backends/MPSGraph.h"
#include "Core/LocalFunction.h"

#include "Kernals/affine_quantization.h"

namespace elman::metal {
Tensor lower_affine_relu(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor codes = inputs[0];
  Tensor zero_points = inputs[2];
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
  return m.select(valid, m.maximum(m.cast(codes, MPSDataTypeInt32), zero), min_code);
}
}  // namespace elman::metal
