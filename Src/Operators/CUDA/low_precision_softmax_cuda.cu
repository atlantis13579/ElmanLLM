#include "kernel.cuh"
#include "Operators/low_precision_softmax.h"
#include "Operators/low_precision_cast.h"
#include "Operators/low_precision_pointwise_math.h"
#include "Core/AllocationPool.h"

namespace elman {
namespace {
__device__ float low_softmax_read(const uint32_t* data, int index, const KernelParams& p) {
    return low_precision_value(data, index, DType(p.input_dtype(KernelInput::A)));
}
__device__ float low_softmax_term(const uint32_t* input, const uint32_t* upstream, int index,
                                const KernelParams& p, bool logarithmic) {
    const float g = low_precision_value(upstream, index, DType(p.output_dtype()));
    return logarithmic ? g : __fmul_rn(g, reinterpret_cast<const float*>(input)[index]);
}
// Each block owns a logical row in FP32 scratch; only the final cast packs bytes.
__global__ void low_softmax_forward_rows(float* result, const uint32_t* input, KernelParams p,
                                        int length, int inner, bool logarithmic, bool persistent) {
    __shared__ float partial[1024];
    const int lane = int(threadIdx.x), width = int(blockDim.x);
    const int base = int(blockIdx.x) / inner * length * inner + int(blockIdx.x) % inner;
    float maximum = -INFINITY;
    for (int j = lane; j < length; j += width) maximum = fmaxf(maximum, low_softmax_read(input, base + j * inner, p));
    if (persistent) {
        for (int offset = width / 2; offset; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(__activemask(), maximum, offset, width));
    } else {
        partial[lane] = maximum; __syncthreads();
        for (int offset = width / 2; offset; offset /= 2) {
            if (lane < offset) partial[lane] = fmaxf(partial[lane], partial[lane + offset]);
            __syncthreads();
        }
        maximum = partial[0];
    }
    float total = 0;
    for (int j = lane; j < length; j += width)
        total += expf(low_softmax_read(input, base + j * inner, p) - maximum);
    if (persistent) {
        for (int offset = width / 2; offset; offset /= 2)
            total += __shfl_xor_sync(__activemask(), total, offset, width);
    } else {
        __syncthreads(); partial[lane] = total; __syncthreads();
        for (int offset = width / 2; offset; offset /= 2) {
            if (lane < offset) partial[lane] += partial[lane + offset];
            __syncthreads();
        }
        total = partial[0];
    }
    const float logsum = logarithmic ? logf(total) : 0;
    for (int j = lane; j < length; j += width) {
        const int i = base + j * inner;
        const float difference = low_softmax_read(input, i, p) - maximum;
        result[i] = logarithmic ? difference - logsum : expf(difference) / total;
    }
}
__global__ void low_softmax_backward_rows(float* result, const uint32_t* input, const uint32_t* upstream,
                                         KernelParams p, int length, int inner, bool logarithmic, int path) {
    __shared__ float partial[1024];
    const int lane = int(threadIdx.x), width = int(blockDim.x);
    const int base = int(blockIdx.x) / inner * length * inner + int(blockIdx.x) % inner;
    float total = 0;
    if (path == 2) {
        // Large contiguous rows follow aligned groups of four with scalar edges.
        const int shift = base % 4;
        int start = base, size = length;
        if (shift) {
            start -= shift; size += shift;
            if (lane >= shift && lane < size) total += low_softmax_term(input, upstream, start + lane, p, logarithmic);
            size -= min(width, size); start += width;
        }
        const int body = size - size % (4 * width);
        for (int offset = lane * 4; offset < body; offset += width * 4)
            for (int j = 0; j < 4; ++j) total += low_softmax_term(input, upstream, start + offset + j, p, logarithmic);
        for (int offset = body + lane; offset < size; offset += width)
            total += low_softmax_term(input, upstream, start + offset, p, logarithmic);
    } else for (int j = lane; j < length; j += width)
        total += low_softmax_term(input, upstream, base + j * inner, p, logarithmic);
    if (path == 0) {
        for (int offset = width / 2; offset; offset /= 2)
            total += __shfl_xor_sync(__activemask(), total, offset, width);
    } else {
        partial[lane] = total; __syncthreads();
        if (path == 2) {
            if (lane == 0) {
                total = 0;
                for (int warp = 0; warp < width / 32; ++warp) {
                    float value = 0;
                    for (int j = 0; j < 32; ++j) value += partial[warp * 32 + j];
                    total += value;
                }
                partial[0] = total;
            }
            __syncthreads();
        } else for (int offset = width / 2; offset; offset /= 2) {
            if (lane < offset) partial[lane] += partial[lane + offset];
            __syncthreads();
        }
        total = partial[0];
    }
    for (int j = lane; j < length; j += width) {
        const int i = base + j * inner;
        const float value = reinterpret_cast<const float*>(input)[i];
        const float term = low_softmax_term(input, upstream, i, p, logarithmic);
        result[i] = __fmaf_rn(-(logarithmic ? expf(value) : value), total, term);
    }
}
}
void launch_softmax_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
void launch_low_precision_softmax_cuda(LowPrecisionSoftmax op, uint32_t* output, const uint32_t* input,
                                      const uint32_t* upstream, const KernelParams& p) {
    if (!p.output_count()) return;
    const size_t bytes = checked_memory_mul(size_t(p.output_count()), sizeof(float));
    auto storage = p.output_dtype() == int(DType::Float32) ? BufferPtr{} : backend_workspace(Backend::CUDA, bytes);
    auto* result = storage ? static_cast<uint32_t*>(storage->raw_pointer()) : output;
    const int axis = p.attribute(0), mode = p.attribute(1);
    const int inner = p.input_rank(KernelInput::A) ? p.input_stride(KernelInput::A, axis) : 1;
    const int length = p.input_rank(KernelInput::A) ? p.input_shape(KernelInput::A, axis) : 1;
    const int rows = p.output_count() / length;
    const bool logarithmic = op == LowPrecisionSoftmax::LogSoftmax;
    int width = 1;
    if (inner == 1) {
        if (mode == 2 && length > 1024) {
            while (width < std::min(length / 4, 1024) / 2) width *= 2;
            width = std::max(width, 32);
        } else while (width < length && width < 32) width *= 2;
    } else if (inner <= 64 && length >= 64) {
        while (inner * width <= 1024 && width <= length) width *= 2;
        width /= 2;
    }
    if (mode == 2) {
        const int path = inner != 1 ? 1 : length <= 1024 ? 0 : 2;
        low_softmax_backward_rows<<<rows, width, 0, current_cuda_stream()>>>(reinterpret_cast<float*>(result), input, upstream, p, length, inner, logarithmic, path);
    } else if (inner == 1 && length > 2048 && !logarithmic) {
        auto params = p; params.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32);
        const bool wide = p.input_dtype(KernelInput::A) == int(DType::Float32);
        auto decoded_storage = wide ? BufferPtr{} : backend_workspace(Backend::CUDA, bytes);
        const uint32_t* decoded = input;
        if (!wide) {
            auto* destination = static_cast<uint32_t*>(decoded_storage->raw_pointer());
            launch_low_precision_cast_cuda(destination, input, params); decoded = destination;
        }
        params.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32);
        launch_softmax_cuda(result, decoded, nullptr, nullptr, params);
    } else {
        const bool persistent = inner == 1 && length <= 2048;
        if (inner == 1 && !persistent) width = 1024;
        low_softmax_forward_rows<<<rows, width, 0, current_cuda_stream()>>>(reinterpret_cast<float*>(result), input, p, length, inner, logarithmic, persistent);
    }
    if (storage) {
        auto params = p;
        params.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32);
        launch_low_precision_cast_cuda(output, result, params);
    }
}
} // namespace elman
