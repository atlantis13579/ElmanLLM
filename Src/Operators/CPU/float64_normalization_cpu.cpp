#include "Operators/float64_normalization_math.h"

namespace elman {
void launch_float64_normalization_cpu(Float64Normalization op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int rows = float64_normalization_rows(op, p);
    for (int row = 0; row < rows; ++row) float64_normalization_row(op, row, reinterpret_cast<double*>(output),
        reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b), reinterpret_cast<const double*>(c), p);
}
} // namespace elman
