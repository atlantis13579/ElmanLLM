#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_maximum(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor lhs = m.float_work(inputs[0]), rhs = m.float_work(inputs[1]);
  if (lhs.dataType != rhs.dataType) {
    MPSDataType type = is_compute_float(DType(p.input_dtype(KernelInput::A))) ||
                        is_compute_float(DType(p.input_dtype(KernelInput::B)))
                    ? MPSDataTypeFloat32
                    : storage_type(DType(p.output_dtype()));
    if (type == MPSDataTypeBool) {
      type = MPSDataTypeInt64;
    }
    lhs = m.cast(lhs, type);
    rhs = m.cast(rhs, type);
  }
  if (is_compute_float(DType(p.input_dtype(KernelInput::A)))) {
    return [m.g maximumWithNaNPropagationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
  }
  return [m.g maximumWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
}  // namespace elman::metal
