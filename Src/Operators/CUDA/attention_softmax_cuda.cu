#include "softmax_rows.cuh"
#include <cstring>

namespace elman {
namespace {
template<int DTypeCode>
struct PaddingMaskInput {
  float inverse;
  int keys, queries, rows_per_batch, mask_batch_stride;
  bool causal;
  int64_t causal_base;

  __device__ float operator()(const uint32_t* scores, const uint32_t* padding, const KernelParams&, int i) const {
    const int row = i / keys, column = i % keys;
    float mask;
    if (causal && int64_t(column) > causal_base + row % queries) {
      mask = DTypeCode == int(DType::Float16) ? -65504.0f :
             DTypeCode == int(DType::BFloat16) ? -3.3895313892515355e38f : -3.402823466e38f;
    } else {
      const int batch_offset = mask_batch_stride ? row / rows_per_batch * mask_batch_stride : 0;
      mask = elman_cuda_decode(elman_cuda_load(padding, batch_offset + column, DTypeCode), DTypeCode);
    }
    const float value = elman_cuda_decode(elman_cuda_load(scores, i, DTypeCode), DTypeCode);
    const float scaled = elman_cuda_decode(elman_cuda_encode(__fmul_rn(value, inverse), DTypeCode), DTypeCode);
    return elman_cuda_decode(elman_cuda_encode(__fadd_rn(scaled, mask), DTypeCode), DTypeCode);
  }
};

template<int DTypeCode>
void launch_padding_softmax(uint32_t* out, const uint32_t* a, const uint32_t* b, const KernelParams& p, float inverse) {
  using namespace operator_attr::attention_softmax;
  const int queries = p.output_shape(2);
  const int mask_batch_stride = p.input_shape(KernelInput::B, 0) == 1 ? 0 : p.input_stride(KernelInput::B, 0);
  const PaddingMaskInput<DTypeCode> read{inverse, p.output_shape(3), queries, p.output_shape(1) * queries,
    mask_batch_stride, p.attribute(kCausal) != 0, int64_t(p.attribute(kQueryBase)) + p.attribute(kCausalOffset)};
  launch_softmax_cuda_impl(out, a, b, p, read, int(DType::Float32));
}
} // namespace

struct AttentionSoftmaxInput {
  float inverse;
  __device__ float operator()(const uint32_t* a, const uint32_t* b, const KernelParams& p, int i) const {
    int dtype = p.input_dtype(KernelInput::A);
    float scaled = __fmul_rn(READ_A(i), inverse);
    scaled = elman_cuda_decode(elman_cuda_encode(scaled, dtype), dtype);
    float mask = READ_B(broadcast_index(i, KernelInput::B, p));
    if (p.attribute(operator_attr::attention_softmax::kCausal)) {
      int keys = p.output_shape(3);
      int64_t query = int64_t((i / keys) % p.output_shape(2)) + p.attribute(operator_attr::attention_softmax::kQueryBase);
      if (int64_t(i % keys) > query + p.attribute(operator_attr::attention_softmax::kCausalOffset))
        mask = dtype == int(DType::Float16) ? -65504.0f : dtype == int(DType::BFloat16) ? -3.3895313892515355e38f : -3.402823466e38f;
    }
    float masked = __fadd_rn(scaled, mask);
    return elman_cuda_decode(elman_cuda_encode(masked, dtype), dtype);
  }
};
void launch_attention_softmax_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                   const uint32_t* c, const KernelParams& p) {
  int bits = p.attribute(operator_attr::attention_softmax::kDivisor);
  float divisor;
  std::memcpy(&divisor, &bits, sizeof(divisor));
  const float inverse = 1.0f / divisor;
  if (p.output_count() && p.output_rank() == 4 && p.attribute(operator_attr::attention_softmax::kAxis) == 3 &&
      p.input_rank(KernelInput::B) == 4 && p.input_shape(KernelInput::B, 1) == 1 &&
      p.input_shape(KernelInput::B, 2) == 1 && p.input_shape(KernelInput::B, 3) == p.output_shape(3) &&
      p.input_stride(KernelInput::B, 3) == 1 && p.input_dtype(KernelInput::A) == p.output_dtype() &&
      p.input_dtype(KernelInput::B) == p.output_dtype()) {
    switch (DType(p.output_dtype())) {
      case DType::Float32: return launch_padding_softmax<int(DType::Float32)>(out, a, b, p, inverse);
      case DType::Float16: return launch_padding_softmax<int(DType::Float16)>(out, a, b, p, inverse);
      case DType::BFloat16: return launch_padding_softmax<int(DType::BFloat16)>(out, a, b, p, inverse);
      default: break;
    }
  }
  // Scalar division uses a host-rounded reciprocal on CUDA. The transformed
  // values enter a Float32 softmax, including its large-row vector traversal.
  launch_softmax_cuda_impl(out, a, b, p, AttentionSoftmaxInput{inverse}, int(DType::Float32));
}
} // namespace elman
