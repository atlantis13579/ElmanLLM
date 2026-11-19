#include "kernel.h"
#include "Operators/low_precision_matrix_math.h"

namespace elman {
void launch_low_precision_matrix_cpu(LowPrecisionMatrix operation, uint32_t* output,
        const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    auto* bytes = reinterpret_cast<uint8_t*>(output);
    const bool packed = dtype == DType::Float4E2M1FN;
    for (int i = 0; i < p.output_count(); i += packed ? 2 : 1) {
        uint8_t code = low_precision_encode(low_precision_matrix_value(operation, i, a, b, c, p), dtype);
        if (packed && i + 1 < p.output_count())
            code |= uint8_t(low_precision_encode(low_precision_matrix_value(operation, i + 1, a, b, c, p), dtype) << 4);
        bytes[packed ? i / 2 : i] = code;
    }
}
} // namespace elman
