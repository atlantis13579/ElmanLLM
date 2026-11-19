#include "indexed_write.cuh"

namespace elman {
void launch_scatter_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  launch_indexed_write<IndexedWrite::Scatter>(out, a, b, c, p);
}
}  // namespace elman
