#include "kernel.cuh"

namespace elman {
__global__ void categorical_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, KernelParams p) {
    int row = int(blockIdx.x), lane = int(threadIdx.x), width = p.input_shape(KernelInput::A, 1);
    float uniform = READ_C(row), total = READ_B(row * width + width - 1);
    int invalid = !isfinite(uniform) || uniform < 0 || uniform >= 1 || !isfinite(total) || total <= 0;
    float threshold = fminf(uniform * total, nextafterf(total, 0.0f));
    int chosen = width, last = -1;
    for (int j = lane; j < width; j += 256) {
        float weight = READ_A(row * width + j), cumulative = READ_B(row * width + j);
        invalid |= !isfinite(weight) || weight < 0 || !isfinite(cumulative) || cumulative < 0;
        if (weight > 0) {
            last = j;
            if (cumulative > threshold) chosen = min(chosen, j);
        }
    }
    __shared__ int errors[256], choices[256], lasts[256];
    errors[lane] = invalid; choices[lane] = chosen; lasts[lane] = last;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (lane < offset) {
            errors[lane] |= errors[lane + offset];
            choices[lane] = min(choices[lane], choices[lane + offset]);
            lasts[lane] = max(lasts[lane], lasts[lane + offset]);
        }
        __syncthreads();
    }
    if (!lane) out[row] = uint32_t(errors[0] ? -1 : choices[0] == width ? lasts[0] : choices[0]);
}
void launch_categorical_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (p.output_count()) categorical_cuda<<<p.output_count(), 256, 0, current_cuda_stream()>>>(out, a, b, c, p);
}
} // namespace elman
