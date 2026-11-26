#pragma once

#include "Core/Backend.h"
#include <cstdint>

#ifdef __CUDACC__
#define ELMAN_QUANT_HD __host__ __device__
#else
#define ELMAN_QUANT_HD
#endif

namespace elman {
enum class AffineQuantization { None, Quantize, Dequantize, Embedding, Relu };

ELMAN_QUANT_HD inline constexpr unsigned quantized_storage_bits(DType dtype) {
    switch (dtype) {
    case DType::QInt8: case DType::QUInt8: return 8;
    case DType::QInt32: return 32;
    case DType::QUInt4: return 4;
    case DType::QUInt2: return 2;
    default: return 0;
    }
}
ELMAN_QUANT_HD inline constexpr int64_t quantized_minimum(DType dtype) {
    return dtype == DType::QInt32 ? -INT64_C(2147483648) : dtype == DType::QInt8 ? -128 : 0;
}
ELMAN_QUANT_HD inline constexpr int64_t quantized_maximum(DType dtype) {
    return dtype == DType::QInt32 ? INT64_C(2147483647) : dtype == DType::QInt8 ? 127 :
           dtype == DType::QUInt8 ? 255 : dtype == DType::QUInt4 ? 15 : 3;
}
ELMAN_QUANT_HD inline int64_t quantized_code(const void* input, uint64_t index, DType dtype) {
    switch (dtype) {
    case DType::QInt32: return static_cast<const int32_t*>(input)[index];
    case DType::QInt8: return static_cast<const int8_t*>(input)[index];
    case DType::QUInt8: return static_cast<const uint8_t*>(input)[index];
    default: {
        const unsigned bits = quantized_storage_bits(dtype);
        const uint64_t start = index * bits;
        return (static_cast<const uint8_t*>(input)[start / 8] >> (start % 8)) & ((1u << bits) - 1u);
    }
    }
}
inline AffineQuantization affine_quantization_operation(const std::string& name) {
    if (name == "affine_relu") return AffineQuantization::Relu;
    if (name == "affine_quantize") return AffineQuantization::Quantize;
    if (name == "affine_dequantize") return AffineQuantization::Dequantize;
    if (name == "affine_embedding") return AffineQuantization::Embedding;
    return AffineQuantization::None;
}
// Metadata broadcasts to [flattened_rows, ceil(columns / group_size)].
ELMAN_QUANT_HD inline int quantization_parameter_index(KernelInput parameter, int row, int group,
                                                       const KernelParams& p) {
    const int rank = p.input_rank(parameter);
    if (rank == 0) return 0;
    if (rank == 1) return p.input_shape(parameter, 0) == 1 ? 0 : group;
    return (p.input_shape(parameter, 0) == 1 ? 0 : row * p.input_stride(parameter, 0)) +
           (p.input_shape(parameter, 1) == 1 ? 0 : group);
}

void validate_affine_quantization(AffineQuantization operation, const BufferPtr& output,
                                  const std::vector<BufferPtr>& inputs, const KernelParams& params);
void launch_affine_quantization_cpu(AffineQuantization operation, uint32_t* output,
                                    const uint32_t* input, const uint32_t* scales,
                                    const uint32_t* zero_points, const uint32_t* indices,
                                    const KernelParams& params);
void launch_affine_quantization_cuda(AffineQuantization operation, uint32_t* output,
                                     const uint32_t* input, const uint32_t* scales,
                                     const uint32_t* zero_points, const uint32_t* indices,
                                     const KernelParams& params);
} // namespace elman
#undef ELMAN_QUANT_HD
