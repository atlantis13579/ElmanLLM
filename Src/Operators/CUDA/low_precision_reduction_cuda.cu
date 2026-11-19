#include "kernel.cuh"
#include "Core/AllocationPool.h"
#include "Operators/low_precision_reduction_math.h"
#include "Operators/low_precision_cast.h"

namespace elman {
void launch_cumsum_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
void launch_cumsum_backward_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
namespace {
struct LowReductionOrder { int width, height; bool split_y, vectorized; };
int low_floor_power_two(int value) {
    int result = 1;
    while (result <= value / 2 && result < 512) result *= 2;
    return result;
}
LowReductionOrder low_reduction_order(const KernelParams& p) {
    int length = 1;
    if (p.attribute(0) >= 0) length = p.input_shape(KernelInput::A, p.attribute(0));
    else for (int d = 0; d < p.input_rank(KernelInput::A); ++d) length *= p.input_shape(KernelInput::A, d);
    const int inner = p.attribute(0) < 0 ? 1 : p.input_stride(KernelInput::A, p.attribute(0));
    const bool fastest = inner == 1, vectorized = fastest && length > 128;
    int output_vector = 1;
    if (!fastest) output_vector = inner % 4 == 0 ? 4 : inner % 2 == 0 ? 2 : 1;
    const int limit = 512 / output_vector;
    const int first = std::min(limit, low_floor_power_two(fastest ? length / (vectorized ? 4 : 1) : p.output_count() / output_vector));
    const int second = std::min(limit, low_floor_power_two(fastest ? p.output_count() : length));
    int width = std::min(first, 32), height = std::min(second, limit / width);
    width = std::min(first, limit / height);
    const int step = fastest ? width : 1;
    return {width, height, (length + step - 1) / step >= std::min(height * 16, 256), vectorized};
}
__device__ int low_reverse_order(int value, int extent) {
    int result = 0;
    for (int bit = 1; bit < extent; bit *= 2) { result = result * 2 + (value & 1); value >>= 1; }
    return result;
}
__device__ float low_reduction_value(LowPrecisionReduction op, int i, const uint32_t* a, const uint32_t* b,
                                     const uint32_t* c, const KernelParams& p, LowReductionOrder order) {
    float value = low_precision_reduction_result(op, i, a, b, c, p);
    if ((op != LowPrecisionReduction::Amax && op != LowPrecisionReduction::Amin) || value != 0.f) return value;
    // Equal signed zeros follow the FP32 reference's four accumulators and
    // lane/warp merge order. Numeric extrema and their gradients are unchanged.
    const int length = low_precision_reduction_length(p), inner = low_precision_reduction_inner(p);
    const int base = p.attribute(0) < 0 ? 0 : i / inner * length * inner + i % inner;
    const bool fastest = inner == 1;
    const int width = fastest ? order.width : 1, height = order.split_y ? order.height : 1;
    const int step = width * height, head = order.vectorized && base % 4 ? 4 - base % 4 : 0;
    const int body = order.vectorized ? (length - head) / 4 * 4 : 0;
    uint64_t selected = 0;
    for (int j = 0; j < length; ++j) {
        const float x = low_precision_value(a, base + j * inner, DType(p.input_dtype(KernelInput::A)));
        if (x != 0.f) continue;
        int thread = j % step, accumulator = (j / step) % 4;
        if (order.vectorized) {
            if (j < head) { thread = j + base % 4; accumulator = 0; }
            else if (j >= head + body) { thread = j - head - body; accumulator = 0; }
            else { thread = ((j - head) / 4) % step; accumulator = (j - head) % 4; }
        }
        const int lane = thread % width, row = thread / width;
        const int groups = width > 32 ? width / 32 : 1;
        const int lane_order = (lane % 32) * groups + low_reverse_order(lane / 32, groups);
        const int rank = ((lane_order * height + low_reverse_order(row, height)) * 4 + accumulator);
        const uint64_t priority = uint64_t(rank) * (uint64_t(length) + 1) + unsigned(j);
        if (priority >= selected) { selected = priority; value = x; }
    }
    return value;
}
__global__ void low_precision_reduction_kernel(LowPrecisionReduction op, uint32_t* out, const uint32_t* a,
                                               const uint32_t* b, const uint32_t* c, KernelParams p, int units, LowReductionOrder order) {
    const int unit = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (unit >= units) return;
    const DType dtype = DType(p.output_dtype());
    if (dtype == DType::Float32) {
        reinterpret_cast<float*>(out)[unit] = low_reduction_value(op, unit, a, b, c, p, order);
    } else if (dtype == DType::Float4E2M1FN) {
        const int i = unit * 2;
        uint8_t code = low_precision_encode(low_reduction_value(op, i, a, b, c, p, order), dtype);
        if (i + 1 < p.output_count())
            code |= uint8_t(low_precision_encode(low_reduction_value(op, i + 1, a, b, c, p, order), dtype) << 4);
        reinterpret_cast<uint8_t*>(out)[unit] = code;
    } else {
        reinterpret_cast<uint8_t*>(out)[unit] = low_precision_encode(low_reduction_value(op, unit, a, b, c, p, order), dtype);
    }
}
}
void launch_low_precision_reduction_cuda(LowPrecisionReduction op, uint32_t* out, const uint32_t* a,
                                         const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int count = p.output_count();
    if (!count) return;
    if (op == LowPrecisionReduction::Cumsum || op == LowPrecisionReduction::ReverseCumsum) {
        // Retain the established FP32 scan order and stream-owned scratch lifetime.
        auto input = backend_workspace(Backend::CUDA, size_t(count) * sizeof(float));
        auto result = backend_workspace(Backend::CUDA, size_t(count) * sizeof(float));
        auto* decoded = static_cast<uint32_t*>(input->raw_pointer());
        auto* scanned = static_cast<uint32_t*>(result->raw_pointer());
        auto params = p;
        params.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32);
        launch_low_precision_cast_cuda(decoded, a, params);
        params.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32);
        if (op == LowPrecisionReduction::Cumsum) launch_cumsum_cuda(scanned, decoded, nullptr, nullptr, params);
        else launch_cumsum_backward_cuda(scanned, decoded, nullptr, nullptr, params);
        params.v[KernelParams::kOutputDTypeSlot] = p.output_dtype();
        launch_low_precision_cast_cuda(out, scanned, params);
        return;
    }
    const int units = p.output_dtype() == int(DType::Float4E2M1FN) ? int((unsigned(count) + 1u) / 2u) : count;
    const auto order = op == LowPrecisionReduction::Amax || op == LowPrecisionReduction::Amin ? low_reduction_order(p) : LowReductionOrder{1, 1, false, false};
    low_precision_reduction_kernel<<<(unsigned(units) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(op, out, a, b, c, p, units, order);
}
} // namespace elman
