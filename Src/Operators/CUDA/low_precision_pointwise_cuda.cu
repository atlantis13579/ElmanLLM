#include "kernel.cuh"
#include "Operators/low_precision_pointwise_math.h"

namespace elman {
__global__ void kernel_low_precision_pointwise_cuda(LowPrecisionPointwise op, uint32_t* output, const uint32_t* a,
                                                   const uint32_t* b, const uint32_t* c, KernelParams p) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
    if (p.output_dtype() == int(DType::Float4E2M1FN)) {
        if (index >= (unsigned(p.output_count()) + 1) / 2) return;
        const uint8_t low = low_precision_pointwise_result(op, int(index * 2), a, b, c, p);
        const uint8_t high = index * 2 + 1 < unsigned(p.output_count()) ? low_precision_pointwise_result(op, int(index * 2 + 1), a, b, c, p) : 0;
        bytes[index] = uint8_t(low | (high << 4));
    } else if (index < unsigned(p.output_count())) bytes[index] = low_precision_pointwise_result(op, int(index), a, b, c, p);
}
void launch_low_precision_pointwise_cuda(LowPrecisionPointwise op, uint32_t* output, const uint32_t* a,
                                         const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const unsigned count = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1) / 2 : unsigned(p.output_count());
    if (count) kernel_low_precision_pointwise_cuda<<<(count + 255) / 256, 256, 0, current_cuda_stream()>>>(op, output, a, b, c, p);
}
} // namespace elman
