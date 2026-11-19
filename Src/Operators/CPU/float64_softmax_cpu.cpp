#include "Operators/float64_softmax_math.h"

namespace elman {
void launch_float64_softmax_cpu(Float64Softmax op, uint32_t* output, const uint32_t* input, const KernelParams& p) {
    if (!p.output_count()) return;
    const int axis = p.attribute(0), rank = p.input_rank(KernelInput::A);
    const int length = rank ? p.input_shape(KernelInput::A, axis) : 1;
    const int inner = rank ? p.input_stride(KernelInput::A, axis) : 1;
    for (int row = 0; row < p.output_count() / length; ++row)
        float64_softmax_row(op, reinterpret_cast<double*>(output), reinterpret_cast<const double*>(input), row, length, inner);
}
} // namespace elman
