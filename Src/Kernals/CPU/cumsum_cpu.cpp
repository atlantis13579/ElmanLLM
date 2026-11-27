#include "kernel.h"
#include "Kernals/scan_params.h"
#include "Kernals/integer_reduction.h"

namespace elman {
namespace {
void scan_cpu(uint32_t* out, const uint32_t* a, const KernelParams& p, bool reverse) {
  if (!p.output_count()) return;
  int axis = p.attribute(kernal_attr::scan::kAxis);
  bool scalar = p.input_rank(KernelInput::A) == 0;
  int length = scalar ? 1 : p.input_shape(KernelInput::A, axis);
  int inner = scalar ? 1 : p.input_stride(KernelInput::A, axis);
  if (p.output_dtype() != int(DType::Int64) && (scalar || (reverse && length == 1))) {
    for (int i = 0; i < p.output_count(); ++i) COPY_INPUT_A(i, i);
    return;
  }
  for (int group = 0; group < p.output_count() / length; ++group) {
    int base = group / inner * length * inner + group % inner;
    double wide = 0;
    float narrow = 0;
    uint64_t integer = 0;
    for (int j = 0; j < length; ++j) {
      int i = base + (reverse ? length - 1 - j : j) * inner;
      if (p.output_dtype() == int(DType::Int64)) {
        uint64_t value = integer_reduction_value(a, i, p.input_dtype(KernelInput::A));
        integer += value;
        reinterpret_cast<uint64_t*>(out)[i] = integer;
      } else if (p.input_dtype(KernelInput::A) == int(DType::Float32)) {
        wide += double(READ_A(i));
        WRITE_OUTPUT(i, float(wide));
      } else {
        narrow += READ_A(i);
        WRITE_OUTPUT(i, narrow);
      }
    }
  }
}
}
void launch_cumsum_cpu(uint32_t* out, const uint32_t* a, const uint32_t*, const uint32_t*, const KernelParams& p) {
  scan_cpu(out, a, p, false);
}
void launch_cumsum_backward_cpu(uint32_t* out, const uint32_t* a, const uint32_t*, const uint32_t*, const KernelParams& p) {
  scan_cpu(out, a, p, true);
}
} // namespace elman
