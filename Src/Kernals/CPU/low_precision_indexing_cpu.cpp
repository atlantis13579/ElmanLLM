#include "kernel.h"
#include "Kernals/low_precision_indexing_math.h"

namespace elman {
void launch_low_precision_indexing_cpu(LowPrecisionIndexing op, uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const unsigned bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1u) / 2u : unsigned(p.output_count());
    for (unsigned byte = 0; byte < bytes; ++byte) {
        const bool packed = p.output_dtype() == int(DType::Float4E2M1FN);
        const int i = packed ? byte * 2 : byte;
        uint8_t value = low_precision_indexing_code(op, i, a, reinterpret_cast<const int32_t*>(b), c, p);
        if (packed && i + 1 < p.output_count()) value |= low_precision_indexing_code(op, i + 1, a, reinterpret_cast<const int32_t*>(b), c, p) << 4;
        reinterpret_cast<uint8_t*>(out)[byte] = value;
    }
}
} // namespace elman
