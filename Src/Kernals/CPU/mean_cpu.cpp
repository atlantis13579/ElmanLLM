#include "Backends/CpuParallel.h"
#include "kernel.h"
#include "row_reduction.h"

namespace elman {
void launch_mean_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  parallel_for_cpu(p.output_count(), 128, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      int axis = p.attribute(operator_attr::reduction::kAxis);
      int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
      int length = axis < 0 ? input_count(KernelInput::A, p)
                            : p.input_shape(KernelInput::A, axis);
      int base = axis < 0 ? 0 : (i / inner) * length * inner + i % inner;
      if (inner == 1 && length >= 8 && p.input_dtype(KernelInput::A) == int(DType::Float32)) {
        WRITE_OUTPUT(i, contiguous_float_sum(length, [&](int j) { return cpu_float(a[base + j]); }) / float(length));
        continue;
      }
      if (length > 128 && p.input_dtype(KernelInput::A) == int(DType::Float32)) {
        float partial[32] = {};
        for (int j = 0; j < length; ++j) partial[j % 32] += READ_A(base + j * inner);
        for (int stride = 16; stride > 0; stride /= 2)
          for (int j = 0; j < stride; ++j) partial[j] += partial[j + stride];
        WRITE_OUTPUT(i, partial[0] / float(length));
        continue;
      }
      float value = 0.0f;
      for (int j = 0; j < length; ++j) {
        float x = READ_A(base + j * inner);
        value = value + x;
      }
      WRITE_OUTPUT(i, value / float(length));
    }
  });
}
}  // namespace elman
