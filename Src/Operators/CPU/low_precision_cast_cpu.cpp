#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Core/LowPrecisionFloat.h"
#include "Operators/low_precision_cast.h"

namespace elman {
void launch_low_precision_cast_cpu(uint32_t* output, const uint32_t* input, const KernelParams& p) {
    const DType source = DType(p.input_dtype(KernelInput::A)), target = DType(p.output_dtype());
    if (source == target) {
        if (output != input) std::memcpy(output, input, dtype_storage_bytes(target, size_t(p.output_count())));
        return;
    }
    LocalFunction read{[&](int i) {
        if (source == DType::Float64) return reinterpret_cast<const double*>(input)[i];
        if (source == DType::Float4E2M1FN) return numeric::float4_to_double(uint8_t(reinterpret_cast<const uint8_t*>(input)[i / 2] >> ((i & 1) * 4)));
        if (is_float8(source)) return double(numeric::float8_to_float(reinterpret_cast<const uint8_t*>(input)[i], source));
        return double(cpu_read(input, i, int(source)));
    }};
    uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
    if (target == DType::Float4E2M1FN) {
        for (int i = 0; i < p.output_count(); i += 2) {
            const uint8_t low = numeric::double_to_float4(read(i));
            const uint8_t high = i + 1 < p.output_count() ? numeric::double_to_float4(read(i + 1)) : 0;
            bytes[i / 2] = uint8_t(low | (high << 4));
        }
    } else for (int i = 0; i < p.output_count(); ++i) {
        const double value = read(i);
        if (is_float8(target)) bytes[i] = numeric::float_to_float8(float(value), target);
        else if (target == DType::Float64) reinterpret_cast<double*>(output)[i] = value;
        else cpu_write(output, i, float(value), int(target));
    }
}
} // namespace elman
