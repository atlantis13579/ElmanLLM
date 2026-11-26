#include "indexed_write.cuh"

namespace elman {
void launch_gather_grad_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  launch_indexed_write<IndexedWrite::GatherGradient>(out, a, b, c, p);
}

void launch_gather_backward_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                 const uint32_t* c, const KernelParams& p) {
  launch_gather_grad_cuda(out, a, b, c, p);
}
}  // namespace elman
