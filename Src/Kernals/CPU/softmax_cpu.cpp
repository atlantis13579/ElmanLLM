#include "kernel.h"
#include "Backends/CpuMath.h"
#include "Core/AllocationPool.h"
#include "Kernals/low_precision_softmax.h"
#include <vector>

namespace elman {
void launch_softmax_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  const int axis = p.attribute(operator_attr::softmax::kAxis);
  const int inner = p.input_rank(KernelInput::A) == 0
                        ? 1
                        : p.input_stride(KernelInput::A, axis);
  const int length = p.input_rank(KernelInput::A) == 0
                         ? 1
                         : p.input_shape(KernelInput::A, axis);
  if (p.output_count() == 0) {
    return;
  }
  if (p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == int(DType::Float32)) {
    if (inner == 1 && try_softmax_math_cpu(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(a), p.output_count() / length, length)) return;
    // Reuse the FP32 reference-order implementation when vector libraries are
    // unavailable or the reduction has an inner stride. Its scalar exponential
    // adaptation and lane reduction retain the Float32 rounding order.
    KernelParams exact = p;
    exact.v[KernelParams::kAttributeSlot + 1] = 1;
    launch_low_precision_softmax_cpu(LowPrecisionSoftmax::Softmax, out, a, nullptr, exact);
    return;
  }
  if (inner == 1 && is_narrow_float(DType(p.input_dtype(KernelInput::A))) && cpu_vector_math_available()) {
    BufferPtr workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(p.output_count()), 2 * sizeof(float)));
    float* input = static_cast<float*>(workspace->raw_pointer()), *output = input + p.output_count();
    for (int i = 0; i < p.output_count(); ++i) input[i] = READ_A(i);
    if (try_softmax_math_cpu(output, input, p.output_count() / length, length, true)) {
      for (int i = 0; i < p.output_count(); ++i) WRITE_OUTPUT(i, output[i]);
      return;
    }
  }
  for (int group = 0; group < p.output_count() / length; ++group) {
    const int base = (group / inner) * length * inner + group % inner;
    float maximum_value = -EL_INF;
    for (int j = 0; j < length; ++j) {
      maximum_value = EL_MAX(maximum_value, READ_A(base + j * inner));
    }
    // Long rows lose small exponential terms with serial Float32 accumulation.
    // The scalar fallback keeps them until the final normalization rounding.
    double denominator = 0.0;
    for (int j = 0; j < length; ++j) {
      denominator += EL_EXP(READ_A(base + j * inner) - maximum_value);
    }
    for (int j = 0; j < length; ++j) {
      const int i = base + j * inner;
      WRITE_OUTPUT(i, float(double(EL_EXP(READ_A(i) - maximum_value)) / denominator));
    }
  }
}
}  // namespace elman
