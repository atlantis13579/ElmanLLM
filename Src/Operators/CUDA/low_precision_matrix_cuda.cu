#include "kernel.cuh"
#include "Operators/low_precision_matrix_math.h"

namespace elman {
__global__ void low_precision_matrix_kernel(LowPrecisionMatrix operation, uint8_t* output,
        const uint32_t* a, const uint32_t* b, const uint32_t* c, KernelParams p, int units) {
    const unsigned unit = blockIdx.x * blockDim.x + threadIdx.x;
    if (unit >= unsigned(units)) return;
    const DType dtype = DType(p.output_dtype());
    const bool packed = dtype == DType::Float4E2M1FN;
    const int i = int(unit) * (packed ? 2 : 1);
    uint8_t code = low_precision_encode(low_precision_matrix_value(operation, i, a, b, c, p), dtype);
    if (packed && i + 1 < p.output_count())
        code |= uint8_t(low_precision_encode(low_precision_matrix_value(operation, i + 1, a, b, c, p), dtype) << 4);
    output[unit] = code;
}
void launch_low_precision_matrix_cuda(LowPrecisionMatrix operation, uint32_t* output,
        const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int units = p.output_dtype() == int(DType::Float4E2M1FN) ? int((unsigned(p.output_count()) + 1) / 2) : p.output_count();
    if (units) low_precision_matrix_kernel<<<(unsigned(units) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        operation, reinterpret_cast<uint8_t*>(output), a, b, c, p, units);
}
} // namespace elman
