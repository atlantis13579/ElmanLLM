#include "kernel.cuh"
#include "Operators/float64_pointwise_math.h"

namespace elman {
namespace {
__device__ double read_double(const uint32_t* data, int index, int dtype) {
    return dtype == int(DType::Float64) ? reinterpret_cast<const double*>(data)[index] :
        double(elman_cuda_decode(elman_cuda_load(data, index, dtype), dtype));
}
__global__ void float64_pointwise_kernel(Float64Pointwise op, uint32_t* out, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c, KernelParams p, int arity) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= p.output_count()) return;
    const double x = read_double(a, broadcast_index(i, KernelInput::A, p), p.input_dtype(KernelInput::A));
    const double y = arity >= 2 ? read_double(b, broadcast_index(i, KernelInput::B, p), p.input_dtype(KernelInput::B)) : 0;
    const double z = arity >= 3 ? read_double(c, broadcast_index(i, KernelInput::C, p), p.input_dtype(KernelInput::C)) : 0;
    double value;
    if (op == Float64Pointwise::Rotary) {
        const int width = p.output_shape(p.output_rank() - 1), half = width / 2;
        const bool first = i % width < half;
        const double rotated = read_double(a, first ? i + half : i - half, p.input_dtype(KernelInput::A)) * (first ? -1. : 1.);
        value = float64_rotary_value(x, y, rotated, z);
    } else value = float64_pointwise_value(op, x, y, z, float64_pointwise_slope(p));
    if (p.output_dtype() == int(DType::Float64)) reinterpret_cast<double*>(out)[i] = value;
    else elman_cuda_store(out, i, elman_cuda_encode(float(value), p.output_dtype()), p.output_dtype());
}
}
void launch_float64_pointwise_cuda(Float64Pointwise op, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (p.output_count()) float64_pointwise_kernel<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>
        (op, out, a, b, c, p, float64_pointwise_arity(op));
}
} // namespace elman
