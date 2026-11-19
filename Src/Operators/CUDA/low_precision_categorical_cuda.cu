#include "kernel.cuh"
#include "Operators/low_precision_categorical.h"
#include "Operators/low_precision_pointwise_math.h"

namespace elman {
namespace {
__global__ void low_categorical_kernel(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, KernelParams p) {
    const int row = int(blockIdx.x), lane = int(threadIdx.x), width = p.input_shape(KernelInput::A, 1);
    const auto weight_dtype = DType(p.input_dtype(KernelInput::A));
    const auto cumulative_dtype = DType(p.input_dtype(KernelInput::B));
    const auto uniform_dtype = DType(p.input_dtype(KernelInput::C));
    const float uniform = low_precision_value(uniforms, row, uniform_dtype);
    const float total = low_precision_value(cumulative, row * width + width - 1, cumulative_dtype);
    int invalid = !isfinite(uniform) || uniform < 0.f || uniform >= 1.f || !isfinite(total) || total <= 0.f;
    const float threshold = fminf(uniform * total, nextafterf(total, 0.f));
    int chosen = width, last = -1;
    for (unsigned j = unsigned(lane); j < unsigned(width); j += 256u) {
        const float weight = low_precision_value(weights, row * width + int(j), weight_dtype);
        const float value = low_precision_value(cumulative, row * width + int(j), cumulative_dtype);
        invalid |= !isfinite(weight) || weight < 0.f || !isfinite(value) || value < 0.f;
        if (weight > 0.f) {
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
}
void launch_low_precision_categorical_cuda(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& p) {
    if (p.output_count()) low_categorical_kernel<<<p.output_count(), 256, 0, current_cuda_stream()>>>(output, weights, cumulative, uniforms, p);
}
} // namespace elman
