#include "Kernals/float64_spatial_math.h"

namespace elman {
void launch_float64_spatial_cpu(Float64Spatial op, uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    for (int i = 0; i < p.output_count(); ++i)
        reinterpret_cast<double*>(out)[i] = float64_spatial_value(op, i, reinterpret_cast<const double*>(a),
            reinterpret_cast<const double*>(b), reinterpret_cast<const double*>(c), p);
}
} // namespace elman
