#pragma once
#include "Kernals/integer_extrema.h"

namespace elman {
template<bool Maximum>
void launch_integer_extrema_cpu(uint32_t* out, const uint32_t* input, const KernelParams& p) {
    const int axis = p.attribute(kernal_attr::reduction::kAxis);
    const int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    const int length = axis < 0 ? input_count(KernelInput::A, p) : p.input_shape(KernelInput::A, axis);
    const int dtype = p.input_dtype(KernelInput::A);
    for (int row = 0; row < p.output_count(); ++row) {
        const int base = axis < 0 ? 0 : (row / inner) * length * inner + row % inner;
        uint64_t value = integer_reduction_value(input, base, dtype);
        for (int j = 1; j < length; ++j)
            value = integer_extremum<Maximum>(value, integer_reduction_value(input, base + j * inner, dtype), dtype);
        integer_write(out, row, dtype, value);
    }
}
} // namespace elman
