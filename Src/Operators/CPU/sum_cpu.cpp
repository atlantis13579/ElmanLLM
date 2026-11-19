#include "Backends/CpuParallel.h"
#include "kernel.h"
#include "Operators/integer_reduction.h"

namespace elman {
void launch_sum_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  parallel_for_cpu(p.output_count(), 128, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      int axis = p.attribute(operator_attr::reduction::kAxis);
      int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
      int length = axis < 0 ? input_count(KernelInput::A, p)
                            : p.input_shape(KernelInput::A, axis);
      int base = axis < 0 ? 0 : (i / inner) * length * inner + i % inner;
      if (p.output_dtype() == int(DType::Int64)) {
        uint64_t value = 0;
        for (int j = 0; j < length; ++j)
          value += integer_reduction_value(a, base + j * inner, p.input_dtype(KernelInput::A));
        reinterpret_cast<uint64_t*>(out)[i] = value;
        continue;
      }
      if (kernel_integer(p.output_dtype())) {
        uint32_t integer_value = 0u;
        for (int j = 0; j < length; ++j) {
          integer_value += uint32_t(READ_INT_A(base + j * inner));
        }
        WRITE_INT_OUTPUT(i, int32_t(integer_value));
        continue;
      }
      float value = 0.0f;
      for (int j = 0; j < length; ++j) {
        float x = READ_A(base + j * inner);
        value = value + x;
      }
      WRITE_OUTPUT(i, value);
    }
  });
}
}  // namespace elman
