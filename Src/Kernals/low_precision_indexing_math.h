#pragma once
#include "Kernals/low_precision_indexing.h"
#include "Kernals/low_precision_pointwise_math.h"

#ifdef __CUDACC__
#define ELMAN_LOW_INDEX_HD __device__
#else
#define ELMAN_LOW_INDEX_HD
#endif
namespace elman {
ELMAN_LOW_INDEX_HD inline uint8_t low_precision_indexing_code(LowPrecisionIndexing op, int i, const uint32_t* a,
                                                              const int32_t* ids, const uint32_t* c, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    if (op == LowPrecisionIndexing::RepeatGradient) {
        int count = 1;
        for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
        float value = 0;
        const int delta = p.input_rank(KernelInput::A) - p.output_rank();
        for (int j = 0; j < count; ++j) {
            int flat = j, target = 0, stride = 1;
            for (int d = p.input_rank(KernelInput::A) - 1; d >= 0; --d) {
                const int coordinate = flat % p.input_shape(KernelInput::A, d);
                flat /= p.input_shape(KernelInput::A, d);
                if (d >= delta) {
                    target += coordinate % p.output_shape(d - delta) * stride;
                    stride *= p.output_shape(d - delta);
                }
            }
            if (target == i) value += low_precision_value(a, j, dtype);
        }
        return low_precision_encode(value, dtype);
    }
    const int axis = p.attribute(0);
    if (op == LowPrecisionIndexing::ScatterGradient) {
        int flat = i, source = 0;
        for (int d = p.output_rank() - 1; d >= 0; --d) {
            const int coordinate = flat % p.output_shape(d);
            flat /= p.output_shape(d);
            source += (d == axis ? ids[i] : coordinate) * p.input_stride(KernelInput::A, d);
        }
        return low_precision_code(a, source, dtype);
    }
    const bool scatter = op == LowPrecisionIndexing::Scatter || op == LowPrecisionIndexing::ScatterAdd;
    const KernelInput source = scatter ? KernelInput::C : KernelInput::A;
    int flat = i, base = 0, selected = 0;
    float value = scatter ? low_precision_value(a, i, dtype) : 0.f;
    uint8_t code = scatter ? low_precision_code(a, i, dtype) : low_precision_encode(0.f, dtype);
    for (int d = p.output_rank() - 1; d >= 0; --d) {
        const int coordinate = flat % p.output_shape(d);
        flat /= p.output_shape(d);
        if (d == axis) selected = coordinate;
        else {
            if (coordinate >= p.input_shape(source, d)) return scatter ? code : low_precision_encode(value, dtype);
            base += coordinate * p.input_stride(source, d);
        }
    }
    for (int k = 0; k < p.input_shape(source, axis); ++k) {
        const int j = base + k * p.input_stride(source, axis);
        if (ids[op == LowPrecisionIndexing::SelectionGradient ? k : j] != selected) continue;
        if (op == LowPrecisionIndexing::SortGradient) return low_precision_code(a, j, dtype);
        if (op == LowPrecisionIndexing::Scatter) code = low_precision_code(c, j, dtype);
        else value += low_precision_value(scatter ? c : a, j, dtype);
    }
    return op == LowPrecisionIndexing::Scatter ? code : low_precision_encode(value, dtype);
}
} // namespace elman
#undef ELMAN_LOW_INDEX_HD
