#include "kernel.cuh"

namespace elman {
__global__ void token_presence_cuda(uint32_t* out, const uint32_t* a, int count, int length, int stride) {
    int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    int id = int32_t(a[i]);
    atomicExch(out + i / length * stride + (id >= 0 && id < stride - 1 ? id : stride - 1), 1u);
}
void launch_token_presence_cuda(uint32_t* out, const uint32_t* a, const uint32_t*, const uint32_t*, const KernelParams& p) {
    if (!p.output_count()) return;
    check_cuda_execution(cudaMemsetAsync(out, 0, size_t(p.output_count()) * sizeof(uint32_t), current_cuda_stream()), "CUDA presence clear");
    int count = p.input_shape(KernelInput::A, 0) * p.input_shape(KernelInput::A, 1);
    if (count) token_presence_cuda<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, a, count,
        p.input_shape(KernelInput::A, 1), p.output_shape(1));
}
} // namespace elman
