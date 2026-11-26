#include "kernel.h"
#include "Backends/CpuMath.h"

namespace elman {
void launch_silu_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  if (p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == int(DType::Float32) &&
      try_unary_math_cpu(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(a), p.output_count(), CpuMathOperation::Silu)) return;
  for (int i = 0; i < p.output_count(); ++i) {
    float x = READ_A(i);
    WRITE_OUTPUT(i, x / (1.0f + EL_EXP(-x)));
  }
}
}  // namespace elman
