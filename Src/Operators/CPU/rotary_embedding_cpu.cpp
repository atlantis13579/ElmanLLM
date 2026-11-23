#include "kernel.h"
#include "Core/LocalFunction.h"

namespace elman {
void launch_rotary_embedding_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                 const uint32_t* c, const KernelParams& p) {
  int width = p.output_shape(p.output_rank() - 1), half = width / 2;
  LocalFunction rounded{[&](float value) {
    if (p.output_dtype() == int(DType::Float16)) return numeric::half_to_float(numeric::float_to_half(value));
    if (p.output_dtype() == int(DType::BFloat16)) return numeric::bfloat_to_float(numeric::float_to_bfloat(value));
    return value;
  }};
  for (int i = 0; i < p.output_count(); ++i) {
    bool first = i % width < half;
    float rotated = first ? -READ_A(i + half) : READ_A(i - half);
    float left = rounded(READ_A(i) * READ_B(broadcast_index(i, KernelInput::B, p)));
    float right = rounded(rotated * READ_C(broadcast_index(i, KernelInput::C, p)));
    WRITE_OUTPUT(i, left + right);
  }
}
} // namespace elman
