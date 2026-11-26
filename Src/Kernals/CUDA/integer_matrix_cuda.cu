#include "Kernals/integer_matrix.h"
#include "Backends/CudaStream.h"
#include <cuda_runtime.h>

namespace elman {
__global__ void integer_matrix_kernel(IntegerMatrix operation, uint32_t* output, const uint32_t* a,
                                      const uint32_t* b, const uint32_t* c, KernelParams p) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < unsigned(p.output_count()))
        integer_write(output, int(i), p.output_dtype(), integer_matrix_value(operation, int(i), a, b, c, p));
}
void launch_integer_matrix_cuda(IntegerMatrix operation, uint32_t* output, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (p.output_count()) integer_matrix_kernel<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        operation, output, a, b, c, p);
}
} // namespace elman
