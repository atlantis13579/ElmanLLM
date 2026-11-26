#include "Backends/CpuParallel.h"
#include "kernel.h"
#include "Backends/CpuGemm.h"
#include "row_reduction.h"

namespace elman {
void launch_linear_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  if (try_gemm_cpu(out, a, b, c, p, CpuGemmOperation::Linear)) return;
  parallel_for_cpu(p.output_count(), 128, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      const int outputs = p.input_shape(KernelInput::B, 0);
      const int inputs = p.input_shape(KernelInput::B, 1);
      const int row = i / outputs, column = i % outputs;
      if (kernel_integer(p.output_dtype())) {
        uint32_t value = 0;
        for (int k = 0; k < inputs; ++k)
          value += uint32_t(READ_INT_A(row * inputs + k)) * uint32_t(READ_INT_B(column * inputs + k));
        if (p.input_rank(KernelInput::C) >= 0) value += uint32_t(READ_INT_C(p.input_rank(KernelInput::C) == 0 ? 0 : column));
        WRITE_INT_OUTPUT(i, int32_t(value));
      } else {
        float value = 0;
        if (is_narrow_float(DType(p.output_dtype())))
          value = contiguous_narrow_dot(inputs, [&](int k) { return READ_A(row * inputs + k); },
                                               [&](int k) { return READ_B(column * inputs + k); });
        else value = contiguous_float_dot(inputs, [&](int k) { return READ_A(row * inputs + k); },
                                                  [&](int k) { return READ_B(column * inputs + k); });
        if (p.input_rank(KernelInput::C) >= 0) value += READ_C(p.input_rank(KernelInput::C) == 0 ? 0 : column);
        WRITE_OUTPUT(i, value);
      }
    }
  });
}
} // namespace elman
