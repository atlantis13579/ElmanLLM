#include "Core/AllocationPool.h"
#include "kernel.cuh"
#include "row_reduce.cuh"

namespace elman {
namespace {
constexpr int threads = 128;

__global__ void weight_gradient_terms(float* terms, const uint32_t* a, const uint32_t* b,
                                      int channels, int spatial, int per_group, KernelParams p) {
    __shared__ float scratch[threads];
    const int channel = int(blockIdx.x);
    const int start = channel * spatial;
    const int length = per_group * spatial;
    const int base = (channel / per_group) * length;
    float sum = 0;
    for (int i = threadIdx.x; i < length; i += threads) sum += READ_A(base + i);
    const float mean = row_reduce<false>(sum, scratch) / float(length);
    float squares = 0, dot = 0;
    for (int i = threadIdx.x; i < length; i += threads) {
        const float delta = READ_A(base + i) - mean;
        squares = fmaf(delta, delta, squares);
    }
    for (int i = threadIdx.x; i < spatial; i += threads)
        dot = fmaf(READ_A(start + i) - mean, READ_B(start + i), dot);
    const float variance = row_reduce<false>(squares, scratch) / float(length);
    dot = row_reduce<false>(dot, scratch);
    if (threadIdx.x == 0)
        terms[channel] = dot * rsqrtf(variance + float_attribute(p, operator_attr::group_norm::kEpsilon));
}

__global__ void finish_weight_gradient(uint32_t* out, const float* terms,
                                       int batches, int channels, KernelParams p) {
    const int channel = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    float sum = 0;
    for (int n = 0; n < batches; ++n) sum += terms[n * channels + channel];
    WRITE_OUTPUT(channel, sum);
}

__device__ float group_weight_term(const uint32_t* a, const uint32_t* b, const KernelParams& p,
                                    int row, int channel, float previous) {
    int channels = p.input_shape(KernelInput::A, 1), spatial = p.input_stride(KernelInput::A, 1);
    if (!spatial) return __int_as_float(0x7fffffff);
    int width = channels / p.attribute(0), length = width * spatial;
    int base = (row * channels + channel / width * width) * spatial;
    float mean = 0, variance = 0;
    for (int j = 0; j < length; ++j) mean = __fadd_rn(mean, READ_A(base + j));
    mean /= float(length);
    for (int j = 0; j < length; ++j) { float delta = READ_A(base + j) - mean; variance = fmaf(delta, delta, variance); }
    float epsilon = float_attribute(p, operator_attr::group_norm::kEpsilon);
    int dtype = p.input_dtype(KernelInput::A);
    if (dtype == 3 || dtype == 4) epsilon = elman_cuda_decode(elman_cuda_encode(epsilon, dtype), dtype);
    float inverse = rsqrtf(variance / float(length) + epsilon);
    if (dtype == 3 || dtype == 4) {
        mean = elman_cuda_decode(elman_cuda_encode(mean, dtype), dtype);
        inverse = elman_cuda_decode(elman_cuda_encode(inverse, dtype), dtype);
    }
    int start = (row * channels + channel) * spatial;
    float centered;
    if (spatial == 1) {
        float dy = READ_B(start);
        centered = fmaf(dy, READ_A(start), -__fmul_rn(dy, mean));
    } else {
        float product = 0, gradient = 0;
        for (int j = 0; j < spatial; ++j) {
            float dy = READ_B(start + j);
            product = __fadd_rn(product, __fmul_rn(dy, READ_A(start + j)));
            gradient = __fadd_rn(gradient, dy);
        }
        centered = fmaf(-gradient, mean, product);
    }
    return fmaf(centered, inverse, previous);
}
__global__ void weight_gradient(uint32_t* out, const uint32_t* a, const uint32_t* b, KernelParams p) {
    unsigned channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= unsigned(p.output_count())) return;
    int batch = p.input_shape(KernelInput::A, 0);
    float value = 0;
    // Preserve the reference's batch reduction lanes without a host fallback.
    if (batch <= 128) {
        for (int row = 0; row < batch; ++row) value = group_weight_term(a, b, p, row, int(channel), value);
    } else {
        float lanes[32]{};
        for (int lane = 0; lane < 32; ++lane)
            for (int row = lane; row < batch; row += 32) lanes[lane] = group_weight_term(a, b, p, row, int(channel), lanes[lane]);
        for (int offset = 16; offset; offset /= 2)
            for (int lane = 0; lane < offset; ++lane) lanes[lane] = __fadd_rn(lanes[lane], lanes[lane + offset]);
        value = lanes[0];
    }
    WRITE_OUTPUT(int(channel), value);
}
}
void launch_group_norm_weight_grad_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                        const uint32_t*, const KernelParams& p) {
    if (!p.output_count()) return;
    const int batch = p.input_shape(KernelInput::A, 0), channels = p.input_shape(KernelInput::A, 1);
    const int spatial = p.input_stride(KernelInput::A, 1);
    const int per_group = channels / p.attribute(operator_attr::group_norm::kGroups);
    const cudaStream_t stream = current_cuda_stream();
    if (batch && spatial >= 128 && p.output_dtype() == int(DType::Float32) &&
        p.input_dtype(KernelInput::A) == int(DType::Float32)) {
        BufferPtr scratch = backend_workspace(Backend::CUDA, size_t(batch) * channels * sizeof(float));
        float* terms = static_cast<float*>(scratch->raw_pointer());
        weight_gradient_terms<<<batch * channels, threads, 0, stream>>>(terms, a, b, channels, spatial, per_group, p);
        finish_weight_gradient<<<(unsigned(channels) + threads - 1) / threads, threads, 0, stream>>>(
            out, terms, batch, channels, p);
        return;
    }
    weight_gradient<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, stream>>>(out, a, b, p);
}
} // namespace elman
