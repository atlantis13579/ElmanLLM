#include "kernel.cuh"
#include "Operators/low_precision_loss_math.h"
#include "Operators/low_precision_softmax.h"
#include "Core/AllocationPool.h"
#include <algorithm>

namespace elman {
namespace {
__global__ void low_mse_forward(uint32_t* output, const uint32_t* a, const uint32_t* b, KernelParams p, int count) {
    const DType dtype = DType(p.output_dtype());
    float total = 0;
    for (int i = 0; i < count; ++i) {
        const float delta = low_precision_value(a, i, dtype) - low_precision_value(b, i, dtype);
        total = __fadd_rn(total, __fmul_rn(delta, delta));
    }
    reinterpret_cast<uint8_t*>(output)[0] = low_precision_encode(total / float(count), dtype);
}
__global__ void low_ce_forward(uint32_t* output, const float* values, const int32_t* ids, KernelParams p) {
    __shared__ float partial[1024];
    const int lane = int(threadIdx.x), rows = p.input_shape(KernelInput::A, 0), columns = p.input_shape(KernelInput::A, 1);
    float value = 0;
    for (int row = lane; row < rows; row += int(blockDim.x)) value -= values[row * columns + ids[row]];
    partial[lane] = value; __syncthreads();
    for (int step = int(blockDim.x) / 2; step; step /= 2) {
        if (lane < step) partial[lane] += partial[lane + step];
        __syncthreads();
    }
    if (!lane) reinterpret_cast<uint8_t*>(output)[0] = low_precision_encode(partial[0] / float(rows), DType(p.output_dtype()));
}
__global__ void low_loss_gradient(uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                                   const float* values, KernelParams p, int count, bool mse) {
    const DType dtype = DType(p.input_dtype(KernelInput::A));
    const bool wide = p.output_dtype() == int(DType::Float32), packed = !wide && dtype == DType::Float4E2M1FN;
    const unsigned byte = blockIdx.x * blockDim.x + threadIdx.x;
    const int first = packed ? int(byte * 2) : int(byte);
    if (first >= count) return;
    uint8_t result = 0;
    for (int k = 0; k < (packed ? 2 : 1) && first + k < count; ++k) {
        const int i = first + k;
        float value;
        if (mse) value = low_mse_derivative(i, a, b, c, p, count);
        else {
            const int columns = p.input_shape(KernelInput::A, 1), rows = p.input_shape(KernelInput::A, 0);
            const float selected = -low_precision_value(c, 0, dtype) / float(rows);
            const float total = __fadd_rn(selected, 0.f);
            const float gradient = i % columns == reinterpret_cast<const int32_t*>(b)[i / columns] ? selected : 0.f;
            value = __fmaf_rn(-expf(values[i]), total, gradient);
        }
        if (wide) reinterpret_cast<float*>(output)[i] = value;
        else result |= low_precision_encode(value, dtype) << (4 * k);
    }
    if (!wide) reinterpret_cast<uint8_t*>(output)[byte] = result;
}
}
void launch_low_precision_loss_cuda(LowPrecisionLoss op, uint32_t* output, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int count = low_loss_count(p);
    if (op == LowPrecisionLoss::MeanSquaredError) {
        low_mse_forward<<<1, 1, 0, current_cuda_stream()>>>(output, a, b, p, count); return;
    }
    const bool mse = op == LowPrecisionLoss::MeanSquaredErrorGradient;
    if (mse && !count) return;
    BufferPtr workspace = mse ? BufferPtr{} : backend_workspace(Backend::CUDA, checked_memory_mul(size_t(count), sizeof(float)));
    float* values = workspace ? static_cast<float*>(workspace->raw_pointer()) : nullptr;
    if (!mse) launch_low_precision_softmax_cuda(LowPrecisionSoftmax::LogSoftmax, reinterpret_cast<uint32_t*>(values), a, nullptr, low_loss_softmax_params(p));
    if (op == LowPrecisionLoss::CrossEntropy) {
        const int rows = p.input_shape(KernelInput::A, 0);
        const int exponent = rows >= 16 ? int(std::round(std::log2(double(rows / 16)))) : 0;
        const int threads = std::clamp(1 << std::min(exponent, 10), 32, 1024);
        low_ce_forward<<<1, threads, 0, current_cuda_stream()>>>(output, values, reinterpret_cast<const int32_t*>(b), p);
    } else if (count) {
        const unsigned bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(count) + 1u) / 2u : unsigned(count);
        low_loss_gradient<<<(bytes + 255u) / 256u, 256, 0, current_cuda_stream()>>>(output, a, b, c, values, p, count, mse);
    }
}
} // namespace elman
