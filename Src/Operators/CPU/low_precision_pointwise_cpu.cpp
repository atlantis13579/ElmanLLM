#include "kernel.h"
#include "Operators/low_precision_pointwise_math.h"

namespace elman {
void launch_low_precision_pointwise_cpu(LowPrecisionPointwise op, uint32_t* output, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
    if (p.output_dtype() == int(DType::Float4E2M1FN)) {
        for (int i = 0; i < p.output_count(); i += 2) {
            const uint8_t low = low_precision_pointwise_result(op, i, a, b, c, p);
            const uint8_t high = i + 1 < p.output_count() ? low_precision_pointwise_result(op, i + 1, a, b, c, p) : 0;
            bytes[i / 2] = uint8_t(low | (high << 4));
        }
    } else for (int i = 0; i < p.output_count(); ++i) bytes[i] = low_precision_pointwise_result(op, i, a, b, c, p);
}
} // namespace elman
