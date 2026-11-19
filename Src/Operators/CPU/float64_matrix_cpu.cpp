#include "Operators/float64_matrix_math.h"

namespace elman {
void launch_float64_matrix_cpu(Float64Matrix operation, uint32_t* output,
        const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    for (int i = 0; i < p.output_count(); ++i)
        reinterpret_cast<double*>(output)[i] = float64_matrix_value(operation, i,
            reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b), reinterpret_cast<const double*>(c), p);
}
} // namespace elman
