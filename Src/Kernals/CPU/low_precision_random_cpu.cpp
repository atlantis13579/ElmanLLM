#include "kernel.h"
#include "Kernals/low_precision_random_math.h"

namespace elman {
void launch_low_precision_random_cpu(LowPrecisionRandom op, uint32_t* output, const uint32_t* input, const KernelParams& p) {
    const bool packed = p.output_dtype() == int(DType::Float4E2M1FN);
    for (int i = 0; i < p.output_count(); i += packed ? 2 : 1) {
        uint8_t code = low_precision_random_code(op, i, input, p);
        if (packed && i + 1 < p.output_count()) code |= uint8_t(low_precision_random_code(op, i + 1, input, p) << 4);
        reinterpret_cast<uint8_t*>(output)[packed ? i / 2 : i] = code;
    }
}
} // namespace elman
