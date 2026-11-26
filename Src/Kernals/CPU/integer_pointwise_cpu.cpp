#include "kernel.h"
#include "Kernals/integer_pointwise.h"

namespace elman {
namespace {
double read_float(const uint32_t* data, int index, int dtype) {
    return dtype == int(DType::Float64) ? reinterpret_cast<const double*>(data)[index] : cpu_read(data, index, dtype);
}
uint64_t float_to_integer(double value, int dtype) {
    if (dtype == int(DType::Bool)) return value != 0;
    // Truncate through a signed integer before narrowing.
    if (dtype == int(DType::Int64) || dtype == int(DType::UInt64)) {
        if (dtype == int(DType::UInt64) && value >= 0x1p63 && value < 0x1p64)
            return uint64_t(value - 0x1p63) + (uint64_t(1) << 63);
        return value >= -0x1p63 && value < 0x1p63 ? uint64_t(int64_t(value)) : uint64_t(1) << 63;
    }
    if (dtype == int(DType::UInt8) || dtype == int(DType::UInt32))
        return value >= -0x1p63 && value < 0x1p63 ? uint64_t(int64_t(value)) : 0;
    return value >= -0x1p31 && value < 0x1p31 ? uint64_t(int64_t(value)) : uint64_t(1) << 31;
}
uint64_t read_value(const uint32_t* data, int index, int dtype) {
    return is_standard_integer(DType(dtype)) ? integer_reduction_value(data, index, dtype) : uint64_t(read_float(data, index, dtype) != 0);
}
}
void launch_integer_pointwise_cpu(IntegerPointwise op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    for (int i = 0; i < p.output_count(); ++i) {
        const int ai = broadcast_index(i, KernelInput::A, p);
        const int ad = p.input_dtype(KernelInput::A), od = p.output_dtype();
        if (op == IntegerPointwise::Cast) {
            if (!is_standard_integer(DType(ad))) integer_write(out, i, od, float_to_integer(read_float(a, ai, ad), od));
            else {
                const uint64_t value = integer_reduction_value(a, ai, ad);
                if (is_standard_integer(DType(od))) integer_write(out, i, od, value);
                else if (od == int(DType::Float64)) reinterpret_cast<double*>(out)[i] = integer_to_float<double>(value, ad);
                else cpu_write(out, i, integer_to_float<float>(value, ad), od);
            }
            continue;
        }
        const uint64_t x = read_value(a, ai, ad);
        const uint64_t y = integer_pointwise_arity(op) >= 2 ? read_value(b, broadcast_index(i, KernelInput::B, p), p.input_dtype(KernelInput::B)) : 0;
        const uint64_t z = integer_pointwise_arity(op) >= 3 ? read_value(c, broadcast_index(i, KernelInput::C, p), p.input_dtype(KernelInput::C)) : 0;
        integer_write(out, i, od, integer_pointwise_value(op, x, y, z, ad));
    }
}
} // namespace elman
