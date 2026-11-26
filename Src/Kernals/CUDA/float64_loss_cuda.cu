#include "Kernals/float64_loss_math.h"
#include "Backends/CudaStream.h"
#include <cuda_runtime.h>

namespace elman {
__global__ void float64_loss_kernel(Float64Loss op, double* output, const double* input,
        const void* b, const double* c, int rows, int columns) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (op == Float64Loss::CrossEntropyGradient) {
        if (row < unsigned(rows)) float64_cross_entropy_gradient(output, input, static_cast<const int32_t*>(b), *c, int(row), rows, columns);
    } else if (row == 0) {
        output[0] = op == Float64Loss::MeanSquaredError ? float64_mean_squared_error(input, static_cast<const double*>(b), rows) :
            float64_cross_entropy(input, static_cast<const int32_t*>(b), rows, columns);
    }
}
void launch_float64_loss_cuda(Float64Loss op, uint32_t* output, const uint32_t* a,
        const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (!p.output_count()) return;
    int rows = p.input_shape(KernelInput::A, 0);
    const int columns = p.input_shape(KernelInput::A, 1);
    if (op == Float64Loss::MeanSquaredError) {
        rows = 1;
        for (int d = 0; d < p.input_rank(KernelInput::A); ++d) rows *= p.input_shape(KernelInput::A, d);
    }
    const unsigned work = op == Float64Loss::CrossEntropyGradient ? unsigned(rows) : 1u;
    float64_loss_kernel<<<(work + 127u) / 128u, 128, 0, current_cuda_stream()>>>(op,
        reinterpret_cast<double*>(output), reinterpret_cast<const double*>(a), b, reinterpret_cast<const double*>(c), rows, columns);
}
} // namespace elman
