#pragma once
#include "Kernals/float64_loss.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_DOUBLE_LOSS_HD __host__ __device__
#else
#define ELMAN_DOUBLE_LOSS_HD
#endif
namespace elman {
ELMAN_DOUBLE_LOSS_HD inline double float64_loss_log_partition(const double* input, int columns, double& maximum) {
    maximum = -INFINITY;
    for (int column = 0; column < columns; ++column) {
        const double value = input[column];
        maximum = maximum != maximum || maximum >= value ? maximum : value;
    }
    double denominator = 0.;
    for (int column = 0; column < columns; ++column) denominator += ::exp(input[column] - maximum);
    return ::log(denominator);
}
ELMAN_DOUBLE_LOSS_HD inline double float64_mean_squared_error(const double* a, const double* b, int count) {
    double total = 0.;
    for (int i = 0; i < count; ++i) {
        const double difference = a[i] - b[i];
        total += difference * difference;
    }
    return total / double(count);
}
ELMAN_DOUBLE_LOSS_HD inline double float64_cross_entropy(const double* input, const int32_t* targets, int rows, int columns) {
    double total = 0.;
    for (int row = 0; row < rows; ++row) {
        const double* values = input + row * columns;
        double maximum;
        const double logarithm = float64_loss_log_partition(values, columns, maximum);
        total -= (values[targets[row]] - maximum) - logarithm;
    }
    return total / double(rows);
}
ELMAN_DOUBLE_LOSS_HD inline void float64_cross_entropy_gradient(double* output, const double* input,
        const int32_t* targets, double upstream, int row, int rows, int columns) {
    const double* values = input + row * columns;
    double maximum;
    const double logarithm = float64_loss_log_partition(values, columns, maximum);
    const double nll_gradient = -upstream / double(rows);
    for (int column = 0; column < columns; ++column) {
        const double logarithm_probability = (values[column] - maximum) - logarithm;
        const double selected = column == targets[row] ? nll_gradient : 0.;
        output[row * columns + column] = selected - ::exp(logarithm_probability) * nll_gradient;
    }
}
} // namespace elman
#undef ELMAN_DOUBLE_LOSS_HD
