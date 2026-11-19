#include "kernel.cuh"
#include "Operators/float64_reduction_math.h"

namespace elman {
void launch_cumsum_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
void launch_cumsum_backward_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
namespace {
__device__ bool prefer_extreme(bool maximum, double x, int ix, double y, int iy) {
    if (iy == INT_MAX) return false;
    if (ix == INT_MAX) return true;
    if (isnan(x) || isnan(y)) return isnan(y) && (!isnan(x) || iy < ix);
    return (maximum ? y > x : y < x) || (x == y && iy < ix);
}
__global__ void double_extrema_rows(bool maximum, double* out, const double* a, int rows, int length) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y), lane = int(threadIdx.x);
    if (row >= rows) return;
    const double* input = a + row * length;
    double best = 0.;
    int index = INT_MAX;
    for (unsigned i = unsigned(lane); i < unsigned(length); i += 32u) {
        const double x = input[i];
        if (prefer_extreme(maximum, best, index, x, int(i))) { best = x; index = int(i); }
    }
    for (int step = 16; step; step >>= 1) {
        const double value = __shfl_down_sync(0xffffffff, best, step);
        const int pos = __shfl_down_sync(0xffffffff, index, step);
        if (prefer_extreme(maximum, best, index, value, pos)) { best = value; index = pos; }
    }
    if (!lane) out[row] = best;
}
__global__ void float64_reduction_kernel(Float64Reduction op, double* out, const double* a,
                                         const double* b, const double* c, KernelParams p, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    const int length = float64_reduction_length(p), inner = float64_reduction_inner(p);
    if (op == Float64Reduction::Gradient) out[i] = float64_reduce_gradient(i, a, b, c, p);
    else out[i] = float64_reduce_row(op, a, p.attribute(0) < 0 ? 0 : i / inner * length * inner + i % inner, length, inner);
}
}
void launch_float64_reduction_cuda(Float64Reduction op, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (op == Float64Reduction::Cumsum) { launch_cumsum_cuda(out, a, b, c, p); return; }
    if (op == Float64Reduction::ReverseCumsum) { launch_cumsum_backward_cuda(out, a, b, c, p); return; }
    const int count = p.output_count();
    if (count && (op == Float64Reduction::Maximum || op == Float64Reduction::Minimum) &&
        float64_reduction_inner(p) == 1 && float64_reduction_length(p) >= 64) {
        double_extrema_rows<<<(unsigned(count) + 3u) / 4u, dim3(32,4), 0, current_cuda_stream()>>>(
            op == Float64Reduction::Maximum, reinterpret_cast<double*>(out), reinterpret_cast<const double*>(a),
            count, float64_reduction_length(p));
        return;
    }
    if (count) float64_reduction_kernel<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>
        (op, reinterpret_cast<double*>(out), reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b),
         reinterpret_cast<const double*>(c), p, count);
}
} // namespace elman
