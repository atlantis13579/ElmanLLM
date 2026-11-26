#pragma once
#include "Kernals/float64_indexing.h"

#ifdef __CUDACC__
#define ELMAN_INDEX_HD __host__ __device__
#else
#define ELMAN_INDEX_HD
#endif
namespace elman {
ELMAN_INDEX_HD inline int float64_sort_target(int i, const int32_t* ids, const KernelParams& p) {
    const int axis = p.attribute(0);
    const int inner = p.input_stride(KernelInput::A, axis);
    const int length = p.input_shape(KernelInput::A, axis);
    return i - (i / inner % length) * inner + ids[i] * inner;
}

ELMAN_INDEX_HD inline double float64_indexing_value(Float64Indexing op, int i, const double* a,
                                                    const int32_t* ids, const double* c, const KernelParams& p) {
    if (op == Float64Indexing::RepeatGradient) {
        int count = 1;
        for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
        double value = 0;
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
            if (target == i) value += a[j];
        }
        return value;
    }
    const int axis = p.attribute(0);
    if (op == Float64Indexing::ScatterGradient) {
        int flat = i, source = 0;
        for (int d = p.output_rank() - 1; d >= 0; --d) {
            const int coordinate = flat % p.output_shape(d);
            flat /= p.output_shape(d);
            source += (d == axis ? ids[i] : coordinate) * p.input_stride(KernelInput::A, d);
        }
        return a[source];
    }
    const bool scatter = op == Float64Indexing::Scatter || op == Float64Indexing::ScatterAdd;
    const KernelInput source = scatter ? KernelInput::C : KernelInput::A;
    int flat = i, base = 0, selected = 0;
    double value = scatter ? a[i] : 0.;
    for (int d = p.output_rank() - 1; d >= 0; --d) {
        const int coordinate = flat % p.output_shape(d);
        flat /= p.output_shape(d);
        if (d == axis) selected = coordinate;
        else {
            if (coordinate >= p.input_shape(source, d)) return value;
            base += coordinate * p.input_stride(source, d);
        }
    }
    for (int k = 0; k < p.input_shape(source, axis); ++k) {
        const int j = base + k * p.input_stride(source, axis);
        if (ids[op == Float64Indexing::SelectionGradient ? k : j] != selected) continue;
        if (op == Float64Indexing::SortGradient) return a[j];
        if (op == Float64Indexing::Scatter) value = c[j];
        else value += scatter ? c[j] : a[j];
    }
    return value;
}
} // namespace elman
#undef ELMAN_INDEX_HD
