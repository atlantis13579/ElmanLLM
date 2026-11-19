#pragma once
#include "Operators/float64_softmax.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_DOUBLE_SOFTMAX_HD __host__ __device__
#else
#define ELMAN_DOUBLE_SOFTMAX_HD
#endif
namespace elman {
ELMAN_DOUBLE_SOFTMAX_HD inline void float64_softmax_row(Float64Softmax op, double* output,
        const double* input, int row, int length, int inner) {
    const int base = row / inner * length * inner + row % inner;
    double maximum = -INFINITY;
    for (int j = 0; j < length; ++j) {
        const double value = input[base + j * inner];
        maximum = maximum != maximum || maximum >= value ? maximum : value;
    }
    double denominator = 0.;
    for (int j = 0; j < length; ++j) denominator += ::exp(input[base + j * inner] - maximum);
    const double logarithm = op == Float64Softmax::LogSoftmax ? ::log(denominator) : 0.;
    for (int j = 0; j < length; ++j) {
        const int index = base + j * inner;
        const double shifted = input[index] - maximum;
        output[index] = op == Float64Softmax::LogSoftmax ? shifted - logarithm : ::exp(shifted) / denominator;
    }
}
} // namespace elman
#undef ELMAN_DOUBLE_SOFTMAX_HD
