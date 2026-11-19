#include "Operators/float64_loss_math.h"

namespace elman {
void launch_float64_loss_cpu(Float64Loss op, uint32_t* output, const uint32_t* a,
        const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    auto destination = reinterpret_cast<double*>(output);
    auto input = reinterpret_cast<const double*>(a);
    if (op == Float64Loss::MeanSquaredError) {
        int count = 1;
        for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
        destination[0] = float64_mean_squared_error(input, reinterpret_cast<const double*>(b), count);
        return;
    }
    const int rows = p.input_shape(KernelInput::A, 0), columns = p.input_shape(KernelInput::A, 1);
    const auto targets = reinterpret_cast<const int32_t*>(b);
    if (op == Float64Loss::CrossEntropy) destination[0] = float64_cross_entropy(input, targets, rows, columns);
    else for (int row = 0; row < rows; ++row)
        float64_cross_entropy_gradient(destination, input, targets, *reinterpret_cast<const double*>(c), row, rows, columns);
}
} // namespace elman
