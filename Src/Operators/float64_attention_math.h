#pragma once
#include "Operators/float64_attention.h"
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_DOUBLE_ATTENTION_HD __host__ __device__
#else
#define ELMAN_DOUBLE_ATTENTION_HD
#endif
namespace elman {
// Slots 5-7 extend the legacy Float32 divisor without changing its raw ABI.
ELMAN_DOUBLE_ATTENTION_HD inline double float64_attention_divisor(const KernelParams& p, int legacy_slot) {
    if (p.attribute(7)) {
        union { uint64_t bits; double value; } scalar;
        scalar.bits = uint64_t(uint32_t(p.attribute(5))) | (uint64_t(uint32_t(p.attribute(6))) << 32);
        return scalar.value;
    }
    union { uint32_t bits; float value; } scalar;
    scalar.bits = uint32_t(p.attribute(legacy_slot));
    return double(scalar.value);
}
ELMAN_DOUBLE_ATTENTION_HD inline int float64_attention_broadcast(int index, KernelInput input, const KernelParams& p) {
    int offset = 0;
    const int delta = p.output_rank() - p.input_rank(input);
    for (int d = p.output_rank() - 1; d >= 0; --d) {
        const int coordinate = index % p.output_shape(d);
        index /= p.output_shape(d);
        if (d >= delta && p.input_shape(input, d - delta) != 1)
            offset += coordinate * p.input_stride(input, d - delta);
    }
    return offset;
}
ELMAN_DOUBLE_ATTENTION_HD inline void float64_attention_softmax_row(double* output, const double* input,
        const double* mask, int row, int length, int inner, const KernelParams& p) {
    const int base = row / inner * length * inner + row % inner;
    const double divisor = float64_attention_divisor(p, 1);
    double maximum = -INFINITY;
    for (int j = 0; j < length; ++j) {
        const int index = base + j * inner;
        const double value = input[index] / divisor + mask[float64_attention_broadcast(index, KernelInput::B, p)];
        maximum = maximum != maximum || maximum >= value ? maximum : value;
    }
    double total = 0.;
    for (int j = 0; j < length; ++j) {
        const int index = base + j * inner;
        total += ::exp((input[index] / divisor + mask[float64_attention_broadcast(index, KernelInput::B, p)]) - maximum);
    }
    for (int j = 0; j < length; ++j) {
        const int index = base + j * inner;
        output[index] = ::exp((input[index] / divisor + mask[float64_attention_broadcast(index, KernelInput::B, p)]) - maximum) / total;
    }
}
struct Float64AttentionShape {
    int batch, heads, queries, keys, dimension, values, groups;
    ELMAN_DOUBLE_ATTENTION_HD explicit Float64AttentionShape(const KernelParams& p)
        : batch(p.input_shape(KernelInput::A, 0)), heads(p.input_shape(KernelInput::A, 1)),
          queries(p.input_shape(KernelInput::A, 2)), keys(p.input_shape(KernelInput::B, 2)),
          dimension(p.input_shape(KernelInput::A, 3)), values(p.input_shape(KernelInput::C, 3)),
          groups(heads / p.input_shape(KernelInput::B, 1)) {}
};
ELMAN_DOUBLE_ATTENTION_HD inline double float64_attention_score(const double* query, const double* key,
        int head, int row, int column, const Float64AttentionShape& s) {
    const int a = (head * s.queries + row) * s.dimension, b = (head / s.groups * s.keys + column) * s.dimension;
    double value = 0.;
    for (int k = 0; k < s.dimension; ++k) value = ::fma(query[a + k], key[b + k], value);
    return value;
}
ELMAN_DOUBLE_ATTENTION_HD inline double float64_attention_mask(const double* mask, int head, int row,
        int column, const Float64AttentionShape& s, const KernelParams& p) {
    if (p.attribute(2) && int64_t(column) > int64_t(row) + p.attribute(3)) return -DBL_MAX;
    const int coordinates[4] = {head / s.heads, head % s.heads, row, column};
    const int rank = p.input_rank(KernelInput::D);
    int index = 0;
    for (int d = 0; d < rank; ++d)
        if (p.input_shape(KernelInput::D, d) != 1) index += coordinates[4 - rank + d] * p.input_stride(KernelInput::D, d);
    return mask[index];
}
ELMAN_DOUBLE_ATTENTION_HD inline void float64_attention_probabilities(double* scores, const double* mask,
        int head, int query, const Float64AttentionShape& s, const KernelParams& p) {
    const double divisor = float64_attention_divisor(p, 0);
    double maximum = -INFINITY;
    for (int k = 0; k < s.keys; ++k) {
        scores[k] = scores[k] / divisor + float64_attention_mask(mask, head, query, k, s, p);
        maximum = maximum != maximum || maximum >= scores[k] ? maximum : scores[k];
    }
    double total = 0.;
    for (int k = 0; k < s.keys; ++k) total += ::exp(scores[k] - maximum);
    for (int k = 0; k < s.keys; ++k) scores[k] = ::exp(scores[k] - maximum) / total;
}
ELMAN_DOUBLE_ATTENTION_HD inline double float64_attention_value(const double* scores, const double* values,
        int head, int column, const Float64AttentionShape& s) {
    const int base = head / s.groups * s.keys * s.values + column;
    double value = 0.;
    for (int k = 0; k < s.keys; ++k) value = ::fma(scores[k], values[base + k * s.values], value);
    return value;
}
} // namespace elman
#undef ELMAN_DOUBLE_ATTENTION_HD
