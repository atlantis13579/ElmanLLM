#include "kernel.cuh"
#include "Kernals/affine_quantization_math.h"

namespace elman {
namespace {
template<int Output>
__global__ void grouped_q8_decode(uint32_t* output, const uint8_t* codes, const float* scales, const int32_t* zeros, int count) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float scale = scales[index >> 5]; const int zero = zeros[index >> 5];
    const float value = isfinite(scale) && scale > 0 && zero >= 0 && zero <= 255 ? float(int(codes[index]) - zero) * scale : 0.f;
    elman_cuda_store(output, int(index), elman_cuda_encode(value, Output), Output);
}
struct AffineReader {
    __device__ double operator()(const uint32_t* values, int index, int dtype) const {
        return dtype == int(DType::Float64) ? reinterpret_cast<const double*>(values)[index] :
            double(elman_cuda_decode(elman_cuda_load(values, index, dtype), dtype));
    }
};
__global__ void affine_kernel(AffineQuantization operation, uint32_t* out, const uint32_t* input,
                               const uint32_t* scales, const uint32_t* zeros, const uint32_t* indices,
                               KernelParams p, int64_t work) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work) return;
    const int i = int(index);
    if ((operation == AffineQuantization::Quantize || operation == AffineQuantization::Relu)) affine_quantize_unit(i, operation, out, input, scales, zeros, p, AffineReader{});
    else {
        const double value = affine_decode_element(i, operation, input, scales, zeros, indices, p, AffineReader{});
        if (p.output_dtype() == int(DType::Float64)) reinterpret_cast<double*>(out)[i] = value;
        else elman_cuda_store(out, i, elman_cuda_encode(float(value), p.output_dtype()), p.output_dtype());
    }
}
}
void launch_affine_quantization_cuda(AffineQuantization operation, uint32_t* out, const uint32_t* input,
                                     const uint32_t* scales, const uint32_t* zeros, const uint32_t* indices,
                                     const KernelParams& p) {
    if (!p.output_count()) return;
    if (operation == AffineQuantization::Dequantize && p.attribute(0) == 32 &&
        p.input_dtype(KernelInput::A) == int(DType::QUInt8) && p.input_dtype(KernelInput::B) == int(DType::Float32) &&
        p.input_dtype(KernelInput::C) == int(DType::Int32) && is_compute_float(DType(p.output_dtype())) &&
        p.input_rank(KernelInput::A) == 2 && p.input_shape(KernelInput::A, 1) % 32 == 0) {
        const int rows = p.input_shape(KernelInput::A, 0), groups = p.input_shape(KernelInput::A, 1) / 32;
        bool contiguous = true;
        for (auto parameter : {KernelInput::B, KernelInput::C})
            contiguous = contiguous && p.input_rank(parameter) == 2 && p.input_shape(parameter, 0) == rows &&
                p.input_shape(parameter, 1) == groups && p.input_stride(parameter, 0) == groups && p.input_stride(parameter, 1) == 1;
        if (contiguous) {
            const unsigned blocks = unsigned((int64_t(p.output_count()) + 255) / 256);
            const auto* codes = reinterpret_cast<const uint8_t*>(input); const auto* factors = reinterpret_cast<const float*>(scales);
            const auto* offsets = reinterpret_cast<const int32_t*>(zeros); const auto stream = current_cuda_stream();
            if (p.output_dtype() == int(DType::Float32)) grouped_q8_decode<int(DType::Float32)><<<blocks, 256, 0, stream>>>(out, codes, factors, offsets, p.output_count());
            else if (p.output_dtype() == int(DType::Float16)) grouped_q8_decode<int(DType::Float16)><<<blocks, 256, 0, stream>>>(out, codes, factors, offsets, p.output_count());
            else grouped_q8_decode<int(DType::BFloat16)><<<blocks, 256, 0, stream>>>(out, codes, factors, offsets, p.output_count());
            return;
        }
    }
    const unsigned bits = quantized_storage_bits(DType(p.output_dtype()));
    const int64_t work = (operation == AffineQuantization::Quantize || operation == AffineQuantization::Relu) && bits < 8 ?
        (int64_t(p.output_count()) * bits + 7) / 8 : p.output_count();
    affine_kernel<<<unsigned((work + 255) / 256), 256, 0, current_cuda_stream()>>>(operation, out, input, scales, zeros, indices, p, work);
}
void launch_affine_quantize_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    launch_affine_quantization_cuda(AffineQuantization::Quantize, out, a, b, c, nullptr, p);
}
void launch_affine_relu_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    launch_affine_quantization_cuda(AffineQuantization::Relu, out, a, b, c, nullptr, p);
}
void launch_affine_dequantize_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    launch_affine_quantization_cuda(AffineQuantization::Dequantize, out, a, b, c, nullptr, p);
}
void launch_affine_embedding_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* d, const KernelParams& p) {
    launch_affine_quantization_cuda(AffineQuantization::Embedding, out, a, b, c, d, p);
}
} // namespace elman
