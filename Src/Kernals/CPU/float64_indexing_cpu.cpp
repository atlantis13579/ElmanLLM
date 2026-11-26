#include "Kernals/float64_indexing_math.h"

namespace elman {
void launch_float64_indexing_cpu(Float64Indexing op, uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (op == Float64Indexing::SortGradient) {
        // Validated permutations give each source a unique destination.
        for (int i = 0; i < p.output_count(); ++i)
            reinterpret_cast<double*>(out)[float64_sort_target(i, reinterpret_cast<const int32_t*>(b), p)] =
                reinterpret_cast<const double*>(a)[i];
        return;
    }
    for (int i = 0; i < p.output_count(); ++i)
        reinterpret_cast<double*>(out)[i] = float64_indexing_value(op, i, reinterpret_cast<const double*>(a),
            reinterpret_cast<const int32_t*>(b), reinterpret_cast<const double*>(c), p);
}
} // namespace elman
