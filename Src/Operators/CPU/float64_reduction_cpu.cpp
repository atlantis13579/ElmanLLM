#include "Operators/float64_reduction_math.h"

namespace elman {
void launch_float64_reduction_cpu(Float64Reduction op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    double* output = reinterpret_cast<double*>(out);
    const double* input = reinterpret_cast<const double*>(a);
    const int length = float64_reduction_length(p), inner = float64_reduction_inner(p);
    if (op == Float64Reduction::Cumsum || op == Float64Reduction::ReverseCumsum) {
        const int rows = length ? p.output_count() / length : 0;
        for (int row = 0; row < rows; ++row) float64_scan_row(output, input, row, length, inner, op == Float64Reduction::ReverseCumsum);
    } else if (op == Float64Reduction::Gradient) {
        for (int i = 0; i < p.output_count(); ++i) output[i] = float64_reduce_gradient(i, input,
            reinterpret_cast<const double*>(b), reinterpret_cast<const double*>(c), p);
    } else {
        for (int i = 0; i < p.output_count(); ++i) output[i] = float64_reduce_row(op, input,
            p.attribute(0) < 0 ? 0 : i / inner * length * inner + i % inner, length, inner);
    }
}
} // namespace elman
