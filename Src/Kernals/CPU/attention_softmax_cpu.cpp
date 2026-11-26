#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Backends/CpuMath.h"
#include "Core/AllocationPool.h"
#include "low_precision_softmax_exp.h"

namespace elman {
void launch_attention_softmax_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                  const uint32_t* c, const KernelParams& p) {
  if (!p.output_count()) return;
  int axis = p.attribute(operator_attr::attention_softmax::kAxis);
  float divisor = float_attribute(p, operator_attr::attention_softmax::kDivisor);
  const bool causal = p.attribute(operator_attr::attention_softmax::kCausal) != 0;
  const int64_t query_offset = int64_t(p.attribute(operator_attr::attention_softmax::kQueryBase)) +
                               p.attribute(operator_attr::attention_softmax::kCausalOffset);
  const float blocked = p.output_dtype() == int(DType::Float16) ? -65504.0f :
                        p.output_dtype() == int(DType::BFloat16) ? -3.3895313892515355e38f : -3.402823466e38f;
  LocalFunction rounded{[&](float value) {
    if (p.output_dtype() == int(DType::Float16)) return numeric::half_to_float(numeric::float_to_half(value));
    if (p.output_dtype() == int(DType::BFloat16)) return numeric::bfloat_to_float(numeric::float_to_bfloat(value));
    return value;
  }};
  LocalFunction read{[&](int i) {
    float mask = READ_B(broadcast_index(i, KernelInput::B, p));
    if (causal) {
      const int keys = p.output_shape(3);
      if (int64_t(i % keys) > (i / keys) % p.output_shape(2) + query_offset) mask = blocked;
    }
    return rounded(rounded(READ_A(i) / divisor) + mask);
  }};
  const int rank = p.input_rank(KernelInput::A);
  const int length = rank ? p.input_shape(KernelInput::A, axis) : 1;
  const int inner = rank ? p.input_stride(KernelInput::A, axis) : 1;
  const bool last = !rank || axis == rank - 1;
  const bool compact_mask = rank == 4 && axis == 3 && inner == 1 && p.input_rank(KernelInput::B) == 4 &&
                            p.input_shape(KernelInput::B, 1) == 1 && p.input_shape(KernelInput::B, 2) == 1 &&
                            p.input_shape(KernelInput::B, 3) == length && p.input_stride(KernelInput::B, 3) == 1;
  const int mask_batch_stride = compact_mask && p.input_shape(KernelInput::B, 0) != 1 ? p.input_stride(KernelInput::B, 0) : 0;
  // Retain the unfused FP32 arithmetic order with one reusable row. Full-tile
  // materialization would violate the fused attention workspace bound.
  BufferPtr workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(length), sizeof(float)));
  float* values = static_cast<float*>(workspace->raw_pointer());
  for (int row = 0; row < p.output_count() / length; ++row) {
    const int base = row / inner * length * inner + row % inner;
    if (compact_mask) {
      const int queries = p.output_shape(2);
      const int mask_base = row / (p.output_shape(1) * queries) * mask_batch_stride;
      const int visible = causal ? int(std::clamp<int64_t>(row % queries + query_offset + 1, 0, length)) : length;
      for (int j = 0; j < visible; ++j) values[j] = rounded(rounded(READ_A(base + j) / divisor) + READ_B(mask_base + j));
      for (int j = visible; j < length; ++j) values[j] = rounded(rounded(READ_A(base + j) / divisor) + blocked);
    } else {
      for (int j = 0; j < length; ++j) values[j] = read(base + j * inner);
    }
    if (inner != 1 || !try_softmax_math_cpu(values, values, 1, length)) {
      const bool vector = last || row % inner < inner / 8 * 8;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < length; ++j)
        maximum = std::isnan(values[j]) || values[j] > maximum ? values[j] : maximum;
      float lanes[8]{}, sum = 0.f;
      for (int j = 0; j < length; ++j) {
        const float shifted = values[j] - maximum;
        values[j] = vector ? low_softmax_vector_exp(shifted) : std::exp(shifted);
        if (last) {
          if (j < 8) lanes[j] = values[j];
          else lanes[j % 8] += values[j];
        } else sum += values[j];
      }
      if (last) {
        sum = lanes[0];
        for (int j = 1; j < std::min(8, length); ++j) sum += lanes[j];
      }
      const float inverse = 1.f / sum;
      for (int j = 0; j < length; ++j) values[j] = last ? values[j] * inverse : values[j] / sum;
    }
    for (int j = 0; j < length; ++j) WRITE_OUTPUT(base + j * inner, values[j]);
  }
}
} // namespace elman
