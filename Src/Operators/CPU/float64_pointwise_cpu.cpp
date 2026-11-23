#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Operators/float64_pointwise_math.h"

namespace elman {
void launch_float64_pointwise_cpu(Float64Pointwise op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    LocalFunction read{[&](const uint32_t* data, int index, KernelInput input) {
        return p.input_dtype(input) == int(DType::Float64) ? reinterpret_cast<const double*>(data)[index] :
            double(cpu_read(data, index, p.input_dtype(input)));
    }};
    const int arity = float64_pointwise_arity(op);
    for (int i = 0; i < p.output_count(); ++i) {
        const double x = read(a, broadcast_index(i, KernelInput::A, p), KernelInput::A);
        const double y = arity >= 2 ? read(b, broadcast_index(i, KernelInput::B, p), KernelInput::B) : 0;
        const double z = arity >= 3 ? read(c, broadcast_index(i, KernelInput::C, p), KernelInput::C) : 0;
        double value;
        if (op == Float64Pointwise::Rotary) {
            const int width = p.output_shape(p.output_rank() - 1), half = width / 2;
            const bool first = i % width < half;
            const double rotated = read(a, first ? i + half : i - half, KernelInput::A) * (first ? -1. : 1.);
            value = float64_rotary_value(x, y, rotated, z);
        } else value = float64_pointwise_value(op, x, y, z, float64_pointwise_slope(p));
        if (p.output_dtype() == int(DType::Float64)) reinterpret_cast<double*>(out)[i] = value;
        else cpu_write(out, i, float(value), p.output_dtype());
    }
}
} // namespace elman
