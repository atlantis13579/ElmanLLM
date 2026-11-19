#include "Operators/float64_matrix_math.h"
#include "Backends/CudaStream.h"
#include "Backends/CudaGemm.h"
#include "Operators/packed_linear_params.h"
#include <cuda_runtime.h>

namespace elman {
__global__ void float64_matrix_kernel(Float64Matrix operation, double* output,
        const double* a, const double* b, const double* c, KernelParams p) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < unsigned(p.output_count()))
        output[index] = float64_matrix_value(operation, int(index), a, b, c, p);
}
void launch_float64_matrix_cuda(Float64Matrix operation, uint32_t* output,
        const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (!p.output_count()) return;
    if (operation == Float64Matrix::PackedLinear) {
        const int shared = p.input_shape(KernelInput::B, 1), columns = p.input_shape(KernelInput::B, 0);
        const int rows = p.output_count() / columns;
        size_t offset = 0;
        // Preserve each projection's GEMM arithmetic and write its final packed
        // segment directly, without materializing an interleaved matrix.
        for (int group = 0; group < p.attribute(0); ++group) {
            const int width = p.attribute(group + 1);
            if (width) launch_float64_matrix_cuda(Float64Matrix::Linear,
                projection_offset(output, offset * rows, sizeof(double)), a,
                projection_offset(b, offset * shared, sizeof(double)),
                p.input_rank(KernelInput::C) < 0 ? nullptr : projection_offset(c, offset, sizeof(double)), projection_params(p, width));
            offset += width;
        }
        return;
    }
    const bool linear = operation == Float64Matrix::Linear;
    const int shared = p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - 1);
    const int columns = linear ? p.input_shape(KernelInput::B, 0) : p.output_shape(p.output_rank() - 1);
    const double beta = operation == Float64Matrix::Addmm ? float64_matrix_coefficient(p, 2) : 1.;
    // Match the reference's vector-bias cuBLASLt path, including its nonfinite
    // epilogue behavior. Scalar/expanded addends retain the general kernel.
    if ((operation == Float64Matrix::Addmm || (linear && p.input_rank(KernelInput::A) > 1)) &&
        beta == 1. && shared > 1 && columns > 1 && p.input_rank(KernelInput::C) == 1) {
        const double alpha = operation == Float64Matrix::Addmm ? float64_matrix_coefficient(p, 0) : 1.;
        if (try_biased_float64_gemm_cuda(reinterpret_cast<double*>(output), reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b),
                reinterpret_cast<const double*>(c), p.output_count() / columns, columns, shared, linear, alpha)) return;
    }
    if (p.output_count()) float64_matrix_kernel<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        operation, reinterpret_cast<double*>(output), reinterpret_cast<const double*>(a),
        reinterpret_cast<const double*>(b), reinterpret_cast<const double*>(c), p);
}
} // namespace elman
