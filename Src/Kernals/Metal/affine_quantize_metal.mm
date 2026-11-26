#include "Backends/MPSGraph.h"
#include "Core/LocalFunction.h"

#include "Kernals/affine_quantization.h"

namespace elman::metal {
Tensor lower_affine_quantize(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor input = inputs[0];
  Tensor scales = inputs[1];
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
  input = m.reshape(input, @[ @(rows), @(columns) ]);
  const DType code_type = DType(p.output_dtype());
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
  const double lo = static_cast<double>(quantized_minimum(code_type));
  const double hi = static_cast<double>(quantized_maximum(code_type));
  Tensor x = m.div(m.cast(input, MPSDataTypeFloat32), scale);
  Tensor rounded = [m.g rintWithTensor:x name:nil];
  Tensor value = m.add(rounded, m.cast(zero, MPSDataTypeFloat32));
  Tensor clamp = [m.g clampWithTensor:value minValueTensor:m.number(lo) maxValueTensor:m.number(hi) name:nil];
  Tensor output = m.select(m.both(valid, [m.g notWithTensor:[m.g isNaNWithTensor:x name:nil] name:nil]), clamp,
                         m.number(lo));
  if (code_type == DType::QInt32) {
    return m.select([m.g greaterThanOrEqualToWithPrimaryTensor:output secondaryTensor:m.number(hi) name:nil],
                    max_code, m.cast(output, MPSDataTypeInt32));
  }
  return m.cast(output, MPSDataTypeInt32);
}
}  // namespace elman::metal
