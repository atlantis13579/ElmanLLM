#include "kernel.h"
#include "Operators/low_precision_fused_math.h"

namespace elman {
void launch_low_precision_fused_cpu(LowPrecisionFused op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const unsigned bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1u) / 2u : unsigned(p.output_count());
    for (unsigned byte = 0; byte < bytes; ++byte) {
        const DType dtype = DType(p.output_dtype());
        const bool packed = dtype == DType::Float4E2M1FN;
        const int i = packed ? int(byte * 2) : int(byte);
        uint8_t value = low_precision_encode(low_precision_fused_value(op, i, a, b, c, p), dtype);
        if (packed && i + 1 < p.output_count()) value |= low_precision_encode(low_precision_fused_value(op, i + 1, a, b, c, p), dtype) << 4;
        reinterpret_cast<uint8_t*>(output)[byte] = value;
    }
}
} // namespace elman
