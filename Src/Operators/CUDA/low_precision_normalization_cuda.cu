#include "kernel.cuh"
#include "Operators/low_precision_normalization_math.h"
#include "Core/AllocationPool.h"
namespace elman {
namespace {
struct LowMoments { float mean, variance, count; };
struct LowBatchPartial { LowMoments moments; float dot, total; };
__device__ LowMoments low_batch_vertical_merge(LowMoments a, LowMoments b) {
    const float factor = 1.f / fmaxf(1.f, a.count + b.count), delta = a.mean - b.mean;
    return {__fmaf_rn(b.mean, b.count, a.mean * a.count) * factor,
            a.variance + __fmaf_rn(delta * delta * b.count * a.count, factor, b.variance), a.count + b.count};
}
__global__ void low_batch_vertical_partial_kernel(LowBatchPartial* partials, const float* stats, const uint32_t* a,
                                                   const uint32_t* g, KernelParams p, int parts, bool backward) {
    __shared__ LowBatchPartial shared[512];
    LowNormLayout l(LowPrecisionNormalization::Batch, p);
    const int row = int(blockIdx.x) / parts, part = int(blockIdx.x) % parts, thread = int(threadIdx.x);
    const int step = int(blockDim.x) * parts, loops = 1 + (l.width - 1) / (step * 4);
    LowMoments values[4]{}; float dots[4]{}, totals[4]{};
    for (int iteration = 0; iteration < loops; ++iteration) for (int k = 0; k < 4; ++k) {
        const int64_t j = int64_t(part) * int(blockDim.x) + thread + (int64_t(iteration) * 4 + k) * step;
        const bool valid = j < l.width;
        const float x = valid ? low_norm_value(a, l.index(row, int(j)), l) : 0.f;
        if (backward) {
            const float gradient = valid ? low_norm_value(g, l.index(row, int(j)), l) : 0.f;
            dots[k] = __fmaf_rn(gradient, x - stats[row * 2], dots[k]); totals[k] += gradient;
        } else {
            values[k].count += valid ? 1.f : 0.f;
            const float delta = x - values[k].mean;
            values[k].mean = __fmaf_rn(delta, valid ? 1.f / values[k].count : 0.f, values[k].mean);
            values[k].variance = __fmaf_rn(delta * (x - values[k].mean), valid ? 1.f : 0.f, values[k].variance);
        }
    }
    for (int k = 1; k < 4; ++k) {
        if (!backward) values[0] = low_batch_vertical_merge(values[0], values[k]);
        dots[0] += dots[k]; totals[0] += totals[k];
    }
    shared[thread] = {values[0], dots[0], totals[0]}; __syncthreads();
    for (int offset = int(blockDim.x) / 2; offset; offset /= 2) {
        if (thread < offset) {
            if (!backward) shared[thread].moments = low_batch_vertical_merge(shared[thread].moments, shared[thread + offset].moments);
            shared[thread].dot += shared[thread + offset].dot; shared[thread].total += shared[thread + offset].total;
        }
        __syncthreads();
    }
    if (!thread) partials[row * parts + part] = shared[0];
}
__global__ void low_batch_vertical_finish_kernel(float* stats, float* coefficients, const LowBatchPartial* partials,
                                                  KernelParams p, int parts, bool backward) {
    __shared__ LowBatchPartial shared[512];
    LowNormLayout l(LowPrecisionNormalization::Batch, p);
    const int row = int(blockIdx.x), thread = int(threadIdx.x);
    LowBatchPartial value{};
    for (int part = thread; part < parts; part += int(blockDim.x)) {
        const LowBatchPartial other = partials[row * parts + part];
        if (!backward) value.moments = low_batch_vertical_merge(value.moments, other.moments);
        value.dot += other.dot; value.total += other.total;
    }
    shared[thread] = value; __syncthreads();
    for (int offset = int(blockDim.x) / 2; offset; offset /= 2) {
        if (thread < offset) {
            if (!backward) shared[thread].moments = low_batch_vertical_merge(shared[thread].moments, shared[thread + offset].moments);
            shared[thread].dot += shared[thread + offset].dot; shared[thread].total += shared[thread + offset].total;
        }
        __syncthreads();
    }
    if (!thread) {
        // One-part launches already performed the complete vertical reduction.
        value = parts == 1 ? partials[row] : shared[0];
        if (backward) { coefficients[row * 2] = value.dot; coefficients[row * 2 + 1] = value.total; }
        else { stats[row * 2] = value.moments.mean; stats[row * 2 + 1] = rsqrtf(value.moments.variance / float(l.width) + l.epsilon); }
    }
}
__device__ LowMoments low_moments_merge(LowMoments a, LowMoments b) {
    if (!a.count) return b;
    if (!b.count) return a;
    const float delta = b.mean - a.mean, count = a.count + b.count, ratio = b.count / count;
    return {__fmaf_rn(delta, ratio, a.mean), __fmaf_rn(delta * delta * a.count, ratio, a.variance + b.variance), count};
}
__device__ LowMoments low_moments_warp(LowMoments value) {
    for (int offset = 16; offset; offset /= 2) {
        LowMoments other{__shfl_down_sync(0xffffffff, value.mean, offset), __shfl_down_sync(0xffffffff, value.variance, offset), __shfl_down_sync(0xffffffff, value.count, offset)};
        value = low_moments_merge(value, other);
    }
    return value;
}
__device__ LowMoments low_layer_vector_merge(LowMoments b, LowMoments a) {
    // The vectorized ATen layer path weights both means and reverses the merge operands.
    const float count = a.count + b.count;
    if (!count) return {0, 0, 0};
    const float factor = 1.f / count, na = a.count * factor, nb = b.count * factor, delta = b.mean - a.mean;
    return {__fmaf_rn(na, a.mean, nb * b.mean), __fmaf_rn(delta * delta * a.count, nb, a.variance + b.variance), count};
}
__global__ void low_layer_vector_statistics_kernel(float* stats, const uint32_t* a, KernelParams p) {
    __shared__ LowMoments partial[4];
    LowNormLayout l(LowPrecisionNormalization::Layer, p);
    const int row = int(blockIdx.x), thread = int(threadIdx.x), lane = thread % 32, warp = thread / 32;
    LowMoments value{0, 0, 0};
    for (int j = thread * 4; j < l.width; j += 512) for (int k = 0; k < 4; ++k) {
        const float x = low_norm_value(a, row * l.width + j + k, l), delta = x - value.mean;
        value.count += 1;
        value.mean = __fmaf_rn(delta, 1.f / value.count, value.mean);
        value.variance = __fmaf_rn(delta, x - value.mean, value.variance);
    }
    for (int offset = 16; offset; offset /= 2) {
        const LowMoments other{__shfl_down_sync(0xffffffff, value.mean, offset), __shfl_down_sync(0xffffffff, value.variance, offset), __shfl_down_sync(0xffffffff, value.count, offset)};
        value = low_layer_vector_merge(value, other);
    }
    if (!lane) partial[warp] = value;
    __syncthreads();
    for (int offset = 2; offset; offset /= 2) {
        if (!lane && warp < offset) partial[warp] = low_layer_vector_merge(partial[warp], partial[warp + offset]);
        __syncthreads();
    }
    if (!thread) { stats[row * 2] = partial[0].mean; stats[row * 2 + 1] = rsqrtf(partial[0].variance / float(l.width) + l.epsilon); }
}
__global__ void low_norm_moments_kernel(float* stats, const uint32_t* input, KernelParams p, LowPrecisionNormalization op) {
    __shared__ LowMoments partial[32];
    LowNormLayout l(op, p); const int row = int(blockIdx.x), thread = int(threadIdx.x);
    LowMoments value{0, 0, 0};
    for (int j = thread; j < l.width; j += int(blockDim.x)) {
        const float x = low_norm_value(input, l.index(row, j), l), delta = x - value.mean;
        value.count += 1;
        value.mean += delta / value.count;
        value.variance = __fmaf_rn(delta, x - value.mean, value.variance);
    }
    value = low_moments_warp(value);
    if (thread % 32 == 0) partial[thread / 32] = value;
    __syncthreads();
    if (thread < 32) {
        value = thread < int(blockDim.x) / 32 ? partial[thread] : LowMoments{0, 0, 0};
        value = low_moments_warp(value);
        if (!thread) { stats[row * 2] = value.mean; stats[row * 2 + 1] = rsqrtf(value.variance / value.count + l.epsilon); }
    }
}
__device__ float low_norm_block_sum(float value, float* partial) {
    const int thread = int(threadIdx.x);
    for (int offset = 16; offset; offset /= 2) value += __shfl_down_sync(0xffffffff, value, offset);
    __syncthreads(); if (thread % 32 == 0) partial[thread / 32] = value;
    __syncthreads();
    value = thread < int(blockDim.x) / 32 ? partial[thread] : 0.f;
    if (thread < 32) for (int offset = 16; offset; offset /= 2) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
__device__ LowMoments low_batch_warp_moments(LowMoments value) {
    for (int offset = 1; offset < 32; offset *= 2) {
        const float mean = __shfl_xor_sync(0xffffffff, value.mean, offset);
        const float variance = __shfl_xor_sync(0xffffffff, value.variance, offset);
        const float count = __shfl_xor_sync(0xffffffff, value.count, offset);
        const float factor = 1.f / fmaxf(1.f, value.count + count), delta = value.mean - mean;
        value.variance += __fmaf_rn(delta * delta * value.count * count, factor, variance);
        value.mean = __fmaf_rn(value.count, value.mean, count * mean) * factor;
        value.count += count;
    }
    return value;
}
__global__ void low_batch_statistics_kernel(float* stats, const uint32_t* a, KernelParams p, int width) {
    __shared__ LowMoments partial[32];
    LowNormLayout l(LowPrecisionNormalization::Batch, p);
    const int row = int(blockIdx.x), thread = int(threadIdx.x), lane = thread % width, batch = thread / width;
    LowMoments value{0, 0, 0};
    for (int n = batch; n < l.width / l.spatial; n += int(blockDim.x) / width)
        for (int j = lane; j < l.spatial; j += width) {
            const float x = low_norm_value(a, l.index(row, n * l.spatial + j), l), delta = x - value.mean;
            value.count += 1;
            value.mean += delta / value.count;
            value.variance = __fmaf_rn(delta, x - value.mean, value.variance);
        }
    value = low_batch_warp_moments(value);
    if (thread % 32 == 0) partial[thread / 32] = value;
    __syncthreads();
    if (thread < 32) {
        value = thread < int(blockDim.x) / 32 ? partial[thread] : LowMoments{0, 0, 0};
        value = low_batch_warp_moments(value);
        if (!thread) { stats[row * 2] = value.mean; stats[row * 2 + 1] = rsqrtf(value.variance / float(l.width) + l.epsilon); }
    }
}
__global__ void low_batch_coefficients_kernel(float* coefficients, const float* stats, const uint32_t* a,
                                               const uint32_t* g, KernelParams p, int width) {
    __shared__ float partial[32];
    LowNormLayout l(LowPrecisionNormalization::Batch, p);
    const int row = int(blockIdx.x), thread = int(threadIdx.x), lane = thread % width, batch = thread / width;
    float sum = 0, dot = 0;
    for (int n = batch; n < l.width / l.spatial; n += int(blockDim.x) / width)
        for (int j = lane; j < l.spatial; j += width) {
            const int i = l.index(row, n * l.spatial + j);
            const float gradient = low_norm_value(g, i, l), centered = low_norm_value(a, i, l) - stats[row * 2];
            sum += gradient; dot += __fmul_rn(gradient, centered);
        }
    sum = low_norm_block_sum(sum, partial); dot = low_norm_block_sum(dot, partial);
    if (!thread) { coefficients[row * 2] = dot; coefficients[row * 2 + 1] = sum; }
}
__global__ void low_layer_coefficients_kernel(float* coefficients, const float* stats, const uint32_t* a,
                                               const uint32_t* w, const uint32_t* g, KernelParams p) {
    __shared__ float partial[32];
    LowNormLayout l(LowPrecisionNormalization::Layer, p); const int row = int(blockIdx.x);
    const float average = stats[row * 2], inverse = stats[row * 2 + 1];
    float sum = 0, dot = 0;
    int j = int(threadIdx.x) * 4;
    for (; j + 3 < l.width; j += int(blockDim.x) * 4) for (int k = 0; k < 4; ++k) {
        const int i = row * l.width + j + k;
        const float gradient = low_norm_value(g, i, l), weight = low_norm_value(w, j + k, l), x = low_norm_value(a, i, l);
        sum = __fmaf_rn(gradient, weight, sum);
        dot = __fmaf_rn(__fmul_rn(gradient, weight) * (x - average), inverse, dot);
    }
    for (; j < l.width; ++j) {
        const int i = row * l.width + j;
        const float gradient = low_norm_value(g, i, l), weight = low_norm_value(w, j, l), x = low_norm_value(a, i, l);
        sum = __fmaf_rn(gradient, weight, sum);
        dot = __fmaf_rn(__fmul_rn(gradient, weight) * (x - average), inverse, dot);
    }
    sum = low_norm_block_sum(sum, partial);
    dot = low_norm_block_sum(dot, partial);
    if (!threadIdx.x) { coefficients[row * 2] = dot; coefficients[row * 2 + 1] = sum; }
}
__device__ float low_rms_reduce(float value, float* shared, int width) {
    const int thread = int(threadIdx.x), lane = thread % width, row = thread / width, height = int(blockDim.x) / width;
    shared[thread] = value; __syncthreads();
    for (int offset = height / 2; offset; offset /= 2) {
        if (row < offset) shared[thread] += shared[thread + offset * width];
        __syncthreads();
    }
    for (int offset = width / 2; offset >= 32; offset /= 2) {
        if (!row && lane < offset) shared[thread] += shared[thread + offset];
        __syncthreads();
    }
    value = shared[thread];
    const int warp_width = min(width, 32);
    for (int offset = 1; offset < warp_width; offset *= 2) value += __shfl_down_sync(__activemask(), value, offset, warp_width);
    return value;
}
struct LowRmsInput {
    const uint32_t* input;
    const uint32_t* weight;
    const uint32_t* gradient;
    const LowNormLayout& layout;
    int base;
    bool product;

    __device__ float operator()(int column) const {
        const float x = low_norm_value(input, base + column, layout);
        if (!product) return __fmul_rn(x, x);
        const float g = low_norm_value(gradient, base + column, layout);
        const float w = low_norm_value(weight, column, layout);
        return __fmul_rn(low_norm_round(__fmul_rn(g, w), layout), x);
    }
};

__global__ void low_rms_statistics_kernel(float* stats, float* coefficients, const uint32_t* a, const uint32_t* w,
                                          const uint32_t* g, KernelParams p, int width) {
    __shared__ float partial[512];
    LowNormLayout l(LowPrecisionNormalization::Rms, p); const int row = int(blockIdx.x), thread = int(threadIdx.x), step = int(blockDim.x);
    float squares[4]{}, products[4]{};
    const int base = row * l.width;
    const bool vectorized = l.width > 128;
    const int shift = vectorized ? base % 4 : 0, head = shift ? 4 - shift : 0;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass && l.mode != 1) break;
        float* values = pass ? products : squares;
        const LowRmsInput read{a, w, g, l, base, pass != 0};
        if (vectorized) {
            if (shift && thread < 4 && thread >= shift) values[0] += read(thread - shift);
            for (int index = thread * 4 + head; index + 3 < l.width; index += step * 4)
                for (int k = 0; k < 4; ++k) values[k] += read(index + k);
            const int tail = head + (l.width - head) / 4 * 4 + thread;
            if (thread < width && tail < l.width) values[0] += read(tail);
        } else for (int j = thread; j < l.width; j += step * 4)
            for (int k = 0; k < 4 && j + k * step < l.width; ++k) values[k] += read(j + k * step);
    }
    const float square = low_rms_reduce(((squares[0] + squares[1]) + squares[2]) + squares[3], partial, width);
    if (!thread) { stats[row * 2] = 0; stats[row * 2 + 1] = rsqrtf(__fadd_rn(__fmul_rn(square, 1.f / float(l.width)), l.epsilon)); }
    if (l.mode == 1) {
        __syncthreads();
        const float dot = low_rms_reduce(((products[0] + products[1]) + products[2]) + products[3], partial, width);
        if (!thread) { coefficients[row * 2] = dot; coefficients[row * 2 + 1] = 0; }
    }
}
__global__ void low_norm_stats_kernel(float* stats, float* coefficients, const uint32_t* a, const uint32_t* b,
                                      const uint32_t* c, KernelParams p, LowPrecisionNormalization op) {
    LowNormLayout l(op, p);
    const int row = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= l.rows || !l.width) return;
    if (op == LowPrecisionNormalization::Batch) low_norm_statistics(a, stats, row, l);
    if (l.mode == 1) low_norm_coefficients(a, b, c, stats, coefficients, row, l);
}
__global__ void low_norm_result_kernel(uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                                       const float* stats, const float* coefficients, KernelParams p, LowPrecisionNormalization op) {
    LowNormLayout l(op, p);
    const int byte = int(blockIdx.x * blockDim.x + threadIdx.x);
    const bool packed = l.dtype == DType::Float4E2M1FN;
    const int i = packed ? byte * 2 : byte;
    if (i >= p.output_count()) return;
    uint8_t value = low_precision_encode(low_norm_result(i, a, b, c, stats, coefficients, l), l.dtype);
    if (packed && i + 1 < p.output_count()) value |= low_precision_encode(low_norm_result(i + 1, a, b, c, stats, coefficients, l), l.dtype) << 4;
    reinterpret_cast<uint8_t*>(output)[byte] = value;
}
}
void launch_low_precision_normalization_cuda(LowPrecisionNormalization op, uint32_t* output, const uint32_t* a,
                                             const uint32_t* b, const uint32_t* c, const KernelParams& params) {
    KernelParams p = params;
    p.v[KernelParams::kAttributeSlot + 8] = low_precision_normalization_mode(op, params);
    op = low_precision_normalization_base(op);
    if (!p.output_count()) return;
    int rows = 1;
    if (op == LowPrecisionNormalization::Layer || op == LowPrecisionNormalization::Rms)
        for (int d = 0; d < p.input_rank(KernelInput::A) - p.attribute(0); ++d) rows *= p.input_shape(KernelInput::A, d);
    else rows = op == LowPrecisionNormalization::Batch ? p.input_shape(KernelInput::A, 1) : p.input_shape(KernelInput::A, 0) * p.attribute(0);
    int batch_height = 1, batch_parts = 1;
    const bool vertical_batch = op == LowPrecisionNormalization::Batch && p.attribute(9) && p.input_stride(KernelInput::A, 1) == 1;
    if (vertical_batch) {
        int (*const power)(int) = [](int value) { int result = 1; while (result <= value / 2) result *= 2; return result; };
        const int width = std::min(power(rows), 32), count = p.input_shape(KernelInput::A, 0);
        batch_height = std::min(power(count / 16 + int(count % 16 != 0)), 512 / width);
        batch_parts = count ? std::min(1 + (count - 1) / (batch_height * 16), 128) : 1;
        if (batch_parts < 8) batch_parts = 1;
    }
    const size_t statistics_bytes = checked_memory_mul(size_t(std::max(rows, 1)), 4 * sizeof(float));
    const size_t partial_bytes = vertical_batch ? checked_memory_mul(checked_memory_mul(size_t(rows), size_t(batch_parts)), sizeof(LowBatchPartial)) : 0;
    BufferPtr storage = backend_workspace(Backend::CUDA, checked_memory_add(statistics_bytes, partial_bytes));
    float* stats = static_cast<float*>(storage->raw_pointer()); float* coefficients = stats + size_t(rows) * 2;
    LowBatchPartial* partials = reinterpret_cast<LowBatchPartial*>(static_cast<uint8_t*>(storage->raw_pointer()) + statistics_bytes);
    if (rows && (op != LowPrecisionNormalization::Batch || p.attribute(9))) {
        int length = 1;
        if (op == LowPrecisionNormalization::Layer || op == LowPrecisionNormalization::Rms)
            for (int d = p.input_rank(KernelInput::A) - p.attribute(0); d < p.input_rank(KernelInput::A); ++d) length *= p.input_shape(KernelInput::A, d);
        else length = op == LowPrecisionNormalization::Group ? p.input_shape(KernelInput::A, 1) / p.attribute(0) * p.input_stride(KernelInput::A, 1) : p.input_shape(KernelInput::A, 0) * p.input_stride(KernelInput::A, 1);
        if (op == LowPrecisionNormalization::Layer || op == LowPrecisionNormalization::Group) {
            const int threads = op == LowPrecisionNormalization::Group && length < 512 ? 32 : 512;
            if (op == LowPrecisionNormalization::Layer && length % 4 == 0 && length <= (1 << 24))
                low_layer_vector_statistics_kernel<<<rows, 128, 0, current_cuda_stream()>>>(stats, a, p);
            else low_norm_moments_kernel<<<rows, threads, 0, current_cuda_stream()>>>(stats, a, p, op);
            if (p.attribute(8) == 1 && op == LowPrecisionNormalization::Layer)
                low_layer_coefficients_kernel<<<rows, 128, 0, current_cuda_stream()>>>(coefficients, stats, a, b, c, p);
            else if (p.attribute(8) == 1)
                low_norm_stats_kernel<<<(unsigned(rows) + 127u) / 128u, 128, 0, current_cuda_stream()>>>(stats, coefficients, a, b, c, p, op);
        } else if (op == LowPrecisionNormalization::Rms) {
            int (*const power)(int) = [](int value) { int n = 1; while (n <= value / 2 && n < 512) n *= 2; return n; };
            const int first = power(length / (length > 128 ? 4 : 1)), second = power(rows);
            int width = std::min(first, 32), height = std::min(second, 512 / width);
            width = std::min(first, 512 / height);
            if ((length + width - 1) / width < std::min(height * 16, 256)) height = 1;
            low_rms_statistics_kernel<<<rows, width * height, 0, current_cuda_stream()>>>(stats, coefficients, a, b, c, p, width);
        } else if (length) {
            if (vertical_batch) {
                low_batch_vertical_partial_kernel<<<rows * batch_parts, batch_height, 0, current_cuda_stream()>>>(partials, stats, a, c, p, batch_parts, false);
                low_batch_vertical_finish_kernel<<<rows, batch_height, 0, current_cuda_stream()>>>(stats, coefficients, partials, p, batch_parts, false);
                if (p.attribute(8)) {
                    low_batch_vertical_partial_kernel<<<rows * batch_parts, batch_height, 0, current_cuda_stream()>>>(partials, stats, a, p.attribute(8) == 1 ? c : b, p, batch_parts, true);
                    low_batch_vertical_finish_kernel<<<rows, batch_height, 0, current_cuda_stream()>>>(stats, coefficients, partials, p, batch_parts, true);
                }
            } else {
                int width = 32;
                while (width < p.input_stride(KernelInput::A, 1) && width < 512) width *= 2;
                low_batch_statistics_kernel<<<rows, 512, 0, current_cuda_stream()>>>(stats, a, p, width);
                if (p.attribute(8)) low_batch_coefficients_kernel<<<rows, 512, 0, current_cuda_stream()>>>(coefficients, stats, a, p.attribute(8) == 1 ? c : b, p, width);
            }
        }
    }
    const unsigned int bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1u) / 2u : unsigned(p.output_count());
    low_norm_result_kernel<<<(bytes + 255u) / 256u, 256, 0, current_cuda_stream()>>>(output, a, b, c, stats, coefficients, p, op);
}
} // namespace elman
