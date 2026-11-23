#include "Operators/float64_random_math.h"

namespace elman {
void launch_float64_random_cpu(Float64Random op, uint32_t* output, const uint32_t* input, const KernelParams& p) {
    double* destination = reinterpret_cast<double*>(output);
    const double* source = reinterpret_cast<const double*>(input);
    for (int i = 0; i < p.output_count(); ++i)
        destination[i] = float64_random_value(op, uint64_t(i), op == Float64Random::Dropout ? source[i] : 0., p);
}
} // namespace elman
