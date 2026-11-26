#include "kernel.cuh"
#include "Kernals/low_precision_fused_math.h"

namespace elman {
namespace {
__global__ void low_fused_kernel(LowPrecisionFused op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, KernelParams p, unsigned bytes) {
    const unsigned byte = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte < bytes) {
        const DType dtype = DType(p.output_dtype());
        const bool packed = dtype == DType::Float4E2M1FN;
        const int i = packed ? int(byte * 2) : int(byte);
        uint8_t value = low_precision_encode(low_precision_fused_value(op, i, a, b, c, p), dtype);
        if (packed && i + 1 < p.output_count()) value |= low_precision_encode(low_precision_fused_value(op, i + 1, a, b, c, p), dtype) << 4;
        reinterpret_cast<uint8_t*>(output)[byte] = value;
    }
}
}
void launch_low_precision_fused_cuda(LowPrecisionFused op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const unsigned bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1u) / 2u : unsigned(p.output_count());
    if (bytes) low_fused_kernel<<<(bytes + 255u) / 256u, 256, 0, current_cuda_stream()>>>(op, output, a, b, c, p, bytes);
}
} // namespace elman
