#include "comparison.cuh"

namespace elman {
void launch_eq_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t*, const KernelParams& p) {
  launch_comparison<Comparison::Equal>(out, a, b, p);
}
} // namespace elman
