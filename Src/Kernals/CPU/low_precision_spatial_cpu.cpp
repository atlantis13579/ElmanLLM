#include "kernel.h"
#include "Kernals/low_precision_spatial_math.h"

namespace elman {
void launch_low_precision_spatial_cpu(LowPrecisionSpatial op, uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const unsigned bytes = p.output_dtype() == int(DType::Float4E2M1FN) ? (unsigned(p.output_count()) + 1u) / 2u : unsigned(p.output_count());
    for (unsigned byte = 0; byte < bytes; ++byte) {
        const DType dtype = DType(p.output_dtype());
        const LowSpatialValues x{a, dtype}, y{b, dtype}, z{c, dtype};
        const bool packed = dtype == DType::Float4E2M1FN;
        const int i = packed ? byte * 2 : byte;
        uint8_t value = low_precision_encode(low_precision_spatial_value(op, i, x, y, z, p), dtype);
        if (packed && i + 1 < p.output_count()) value |= low_precision_encode(low_precision_spatial_value(op, i + 1, x, y, z, p), dtype) << 4;
        reinterpret_cast<uint8_t*>(out)[byte] = value;
    }
}
} // namespace elman
