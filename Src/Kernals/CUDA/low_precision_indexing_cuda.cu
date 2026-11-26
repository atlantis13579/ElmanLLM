#include "kernel.cuh"
#include "Kernals/low_precision_indexing_math.h"

namespace elman {
namespace {
__global__ void low_precision_indexing_kernel(LowPrecisionIndexing op, uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, KernelParams p, unsigned bytes) {
    const unsigned byte = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte < bytes) {
        const bool packed = p.output_dtype() == int(DType::Float4E2M1FN);
        const int i = packed ? byte * 2 : byte;
        uint8_t value = low_precision_indexing_code(op, i, a, reinterpret_cast<const int32_t*>(b), c, p);
        if (packed && i + 1 < p.output_count()) value |= low_precision_indexing_code(op, i + 1, a, reinterpret_cast<const int32_t*>(b), c, p) << 4;
        reinterpret_cast<uint8_t*>(out)[byte] = value;
    }
}
}
void launch_low_precision_indexing_cuda(LowPrecisionIndexing op, uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const unsigned bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1u) / 2u : unsigned(p.output_count());
    if (bytes) low_precision_indexing_kernel<<<(bytes + 255u) / 256u, 256, 0, current_cuda_stream()>>>(op, out, a, b, c, p, bytes);
}
} // namespace elman
