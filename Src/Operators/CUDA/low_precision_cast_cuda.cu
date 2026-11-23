#include "kernel.cuh"
#include "Core/LowPrecisionFloat.h"
#include "Operators/low_precision_cast.h"

namespace elman {
__device__ double low_precision_read_cuda(const uint32_t* input, int i, DType dtype) {
    if (dtype == DType::Float64) return reinterpret_cast<const double*>(input)[i];
    if (dtype == DType::Float4E2M1FN) return numeric::float4_to_double(uint8_t(reinterpret_cast<const uint8_t*>(input)[i / 2] >> ((i & 1) * 4)));
    if (is_float8(dtype)) return double(numeric::float8_to_float(reinterpret_cast<const uint8_t*>(input)[i], dtype));
    const uint32_t bits = elman_cuda_load(input, i, int(dtype));
    double value = double(elman_cuda_decode(bits, int(dtype)));
    if (value != value) {
        // Preserve the original sign for FP4's explicit signed-zero NaN policy.
        union { uint64_t bits; double value; } nan;
        const unsigned sign_bit = is_narrow_float(dtype) ? 15 : 31;
        nan.bits = 0x7ff8000000000000ull | (uint64_t((bits >> sign_bit) & 1) << 63);
        value = nan.value;
    }
    return value;
}
__global__ void kernel_low_precision_cast_cuda(uint32_t* output, const uint32_t* input, KernelParams p) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const DType source = DType(p.input_dtype(KernelInput::A)), target = DType(p.output_dtype());
    uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
    if (source == target) {
        const unsigned count = target == DType::Float4E2M1FN ? (unsigned(p.output_count()) + 1) / 2 : unsigned(p.output_count());
        if (i < count) bytes[i] = reinterpret_cast<const uint8_t*>(input)[i];
    } else if (target == DType::Float4E2M1FN) {
        if (i >= (unsigned(p.output_count()) + 1) / 2) return;
        const uint8_t low = numeric::double_to_float4(low_precision_read_cuda(input, int(i * 2), source));
        const uint8_t high = i * 2 + 1 < unsigned(p.output_count()) ? numeric::double_to_float4(low_precision_read_cuda(input, int(i * 2 + 1), source)) : 0;
        bytes[i] = uint8_t(low | (high << 4));
    } else if (i < unsigned(p.output_count())) {
        const double value = low_precision_read_cuda(input, int(i), source);
        if (is_float8(target)) bytes[i] = numeric::float_to_float8(float(value), target);
        else if (target == DType::Float64) reinterpret_cast<double*>(output)[i] = value;
        else elman_cuda_store(output, int(i), elman_cuda_encode(float(value), int(target)), int(target));
    }
}
void launch_low_precision_cast_cuda(uint32_t* output, const uint32_t* input, const KernelParams& p) {
    const unsigned count = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1) / 2 : unsigned(p.output_count());
    if (count) kernel_low_precision_cast_cuda<<<(count + 255) / 256, 256, 0, current_cuda_stream()>>>(output, input, p);
}
} // namespace elman
