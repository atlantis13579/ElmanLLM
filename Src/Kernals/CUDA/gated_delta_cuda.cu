#include "kernel.cuh"

namespace elman {
namespace {
__global__ void recurrent(float* out, const float* a, const float* b, const float* c,
                           int batch, int time, int heads, int width, int columns) {
    const int group = int(blockIdx.x), column = int(blockIdx.y * blockDim.x + threadIdx.x);
    if (column >= columns) return;
    float* state = out + batch * time * heads * columns + group * width * columns;
    for (int d = 0; d < width; ++d) state[d * columns + column] = c[(group * width + d) * columns + column];
    for (int t = 0; t < time; ++t) {
        const int row = (group / heads * time + t) * heads + group % heads;
        const float* q = a + row * (2 * width + columns); const float* k = q + width; const float* v = k + width;
        const float decay = b[row * 2], beta = b[row * 2 + 1];
        float prediction = 0;
        for (int d = 0; d < width; ++d) {
            state[d * columns + column] *= decay;
            prediction += state[d * columns + column] * k[d];
        }
        const float delta = (v[column] - prediction) * beta;
        float value = 0;
        for (int d = 0; d < width; ++d) {
            state[d * columns + column] += k[d] * delta;
            value += state[d * columns + column] * q[d];
        }
        out[row * columns + column] = value;
    }
}
}
void launch_gated_delta_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int batch = p.input_shape(KernelInput::A, 0), time = p.input_shape(KernelInput::A, 1);
    const int heads = p.input_shape(KernelInput::A, 2), width = p.attribute(0), columns = p.attribute(1);
    recurrent<<<dim3(batch * heads, (columns + 255) / 256), 256, 0, current_cuda_stream()>>>(reinterpret_cast<float*>(out),
        reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b), reinterpret_cast<const float*>(c),
        batch, time, heads, width, columns);
}
} // namespace elman
