#include "Kernals/float64_categorical.h"
#include "Backends/CudaStream.h"
#include <cuda_runtime.h>

namespace elman {
__global__ void float64_categorical_kernel(uint32_t* output, const double* a, const double* b, const double* c, KernelParams p) {
    const int row = int(blockIdx.x), lane = int(threadIdx.x), width = p.input_shape(KernelInput::A, 1);
    const double uniform = c[row], total = b[row * width + width - 1];
    int invalid = !isfinite(uniform) || uniform < 0. || uniform >= 1. || !isfinite(total) || total <= 0.;
    const double threshold = fmin(uniform * total, nextafter(total, 0.));
    int chosen = width, last = -1;
    for (unsigned j = unsigned(lane); j < unsigned(width); j += 256u) {
        const double weight = a[row * width + j], value = b[row * width + j];
        invalid |= !isfinite(weight) || weight < 0. || !isfinite(value) || value < 0.;
        if (weight > 0.) {
            last = int(j);
            if (value > threshold) chosen = min(chosen, int(j));
        }
    }
    __shared__ int errors[256], choices[256], lasts[256];
    errors[lane] = invalid; choices[lane] = chosen; lasts[lane] = last;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (lane < offset) {
            errors[lane] |= errors[lane + offset];
            choices[lane] = min(choices[lane], choices[lane + offset]);
            lasts[lane] = max(lasts[lane], lasts[lane + offset]);
        }
        __syncthreads();
    }
    if (lane == 0) output[row] = uint32_t(errors[0] ? -1 : choices[0] == width ? lasts[0] : choices[0]);
}
void launch_float64_categorical_cuda(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& p) {
    if (p.output_count()) float64_categorical_kernel<<<p.output_count(), 256, 0, current_cuda_stream()>>>(output,
        reinterpret_cast<const double*>(weights), reinterpret_cast<const double*>(cumulative), reinterpret_cast<const double*>(uniforms), p);
}
} // namespace elman
