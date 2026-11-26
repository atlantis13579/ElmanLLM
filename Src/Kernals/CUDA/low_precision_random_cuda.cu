#include "kernel.cuh"
#include "Kernals/low_precision_random_math.h"

namespace elman {
namespace {
__global__ void low_random_kernel(LowPrecisionRandom op, uint32_t* output, const uint32_t* input, KernelParams p, unsigned bytes) {
    const unsigned byte = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte >= bytes) return;
    const bool packed = p.output_dtype() == int(DType::Float4E2M1FN);
    const int i = packed ? int(byte * 2) : int(byte);
    uint8_t code = low_precision_random_code(op, i, input, p);
    if (packed && i + 1 < p.output_count()) code |= uint8_t(low_precision_random_code(op, i + 1, input, p) << 4);
    reinterpret_cast<uint8_t*>(output)[byte] = code;
}
}
void launch_low_precision_random_cuda(LowPrecisionRandom op, uint32_t* output, const uint32_t* input, const KernelParams& p) {
    const unsigned bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1u) / 2u : unsigned(p.output_count());
    if (bytes) low_random_kernel<<<(bytes + 255u) / 256u, 256, 0, current_cuda_stream()>>>(op, output, input, p, bytes);
}
} // namespace elman
