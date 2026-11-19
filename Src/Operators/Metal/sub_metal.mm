#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_sub(Graph m, const Operands& inputs, const KernelParams& p) {
  auto lhs = m.float_work(inputs[0]), rhs = m.float_work(inputs[1]);
  if (lhs.dataType != rhs.dataType) {
    auto type = is_compute_float(DType(p.input_dtype(KernelInput::A))) ||
                        is_compute_float(DType(p.input_dtype(KernelInput::B)))
                    ? MPSDataTypeFloat32
                    : storage_type(DType(p.output_dtype()));
    if (type == MPSDataTypeBool) {
      type = MPSDataTypeInt64;
    }
    lhs = m.cast(lhs, type);
    rhs = m.cast(rhs, type);
  }
  return [m.g subtractionWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
}  // namespace elman::metal
