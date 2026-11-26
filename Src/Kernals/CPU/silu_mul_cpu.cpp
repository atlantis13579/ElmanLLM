#include "kernel.h"
#include "Backends/CpuMath.h"
#include "Core/AllocationPool.h"

namespace elman {
namespace {
bool matches_output(const KernelParams& p, KernelInput input) {
  const int rank = p.output_rank();
  if (p.input_rank(input) != rank || p.input_dtype(input) != p.output_dtype()) return false;
  int stride = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    if (p.input_shape(input, axis) != p.output_shape(axis) || p.input_stride(input, axis) != stride) return false;
    stride *= p.output_shape(axis);
  }
  return true;
}

template<DType Type>
void contiguous_silu_mul(uint32_t* out, const uint32_t* a, const uint32_t* b, int count) {
  constexpr int dtype = int(Type);
  if constexpr (Type == DType::Float32) {
    if (out != b && try_unary_math_cpu(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(a),
                                      count, CpuMathOperation::Silu)) {
      for (int i = 0; i < count; ++i) out[i] = cpu_bits(cpu_float(out[i]) * cpu_float(b[i]));
      return;
    }
  }
  for (int i = 0; i < count; ++i) {
    const float x = cpu_read(a, i, dtype);
    float activation = x / (1.f + EL_EXP(-x));
    if constexpr (Type == DType::Float16) activation = numeric::half_to_float(numeric::float_to_half(activation));
    if constexpr (Type == DType::BFloat16) activation = numeric::bfloat_to_float(numeric::float_to_bfloat(activation));
    cpu_write(out, i, activation * cpu_read(b, i, dtype), dtype);
  }
}
} // namespace

void launch_silu_mul_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t*, const KernelParams& p) {
  if (!p.output_count()) return;
  if (matches_output(p, KernelInput::A) && matches_output(p, KernelInput::B)) {
    switch (DType(p.output_dtype())) {
      case DType::Float32: return contiguous_silu_mul<DType::Float32>(out, a, b, p.output_count());
      case DType::Float16: return contiguous_silu_mul<DType::Float16>(out, a, b, p.output_count());
      case DType::BFloat16: return contiguous_silu_mul<DType::BFloat16>(out, a, b, p.output_count());
      default: break;
    }
  }
  if (p.output_dtype() == int(DType::Float32) && input_count(KernelInput::A, p) == p.output_count() &&
      try_unary_math_cpu(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(a), p.output_count(), CpuMathOperation::Silu)) {
    for (int i = 0; i < p.output_count(); ++i)
      out[i] = cpu_bits(cpu_float(out[i]) * READ_B(broadcast_index(i, KernelInput::B, p)));
    return;
  }
  BufferPtr workspace;
  const float* activations = nullptr;
  if (p.output_dtype() == int(DType::Float32) && cpu_vector_math_available()) {
    int count = input_count(KernelInput::A, p);
    workspace = backend_workspace(Backend::CPU, size_t(count) * sizeof(float));
    float* values = static_cast<float*>(workspace->raw_pointer());
    if (try_unary_math_cpu(values, reinterpret_cast<const float*>(a), count, CpuMathOperation::Silu)) activations = values;
  }
  for (int i = 0; i < p.output_count(); ++i) {
    int index = broadcast_index(i, KernelInput::A, p);
    float x = READ_A(index);
    float activation = activations ? activations[index] : x / (1.0f + EL_EXP(-x));
    if (p.output_dtype() == int(DType::Float16)) activation = numeric::half_to_float(numeric::float_to_half(activation));
    if (p.output_dtype() == int(DType::BFloat16)) activation = numeric::bfloat_to_float(numeric::float_to_bfloat(activation));
    WRITE_OUTPUT(i, activation * READ_B(broadcast_index(i, KernelInput::B, p)));
  }
}
} // namespace elman
