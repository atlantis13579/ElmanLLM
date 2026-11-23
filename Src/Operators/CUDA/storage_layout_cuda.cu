#include "kernel.cuh"
#include "Operators/storage_layout.h"

namespace elman {
__global__ void kernel_storage_layout(StorageLayout op, uint8_t* output, uint64_t bytes, unsigned bits,
                                      const uint32_t* a, const uint32_t* b, const uint32_t* c, KernelParams p) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) output[i] = storage_layout_byte(op, i, bits, a, b, c, p);
}
void launch_storage_layout_cuda(StorageLayout op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    const size_t bytes = dtype_storage_bytes(dtype, size_t(p.output_count()));
    if (bytes) kernel_storage_layout<<<unsigned((bytes + 255) / 256), 256, 0, current_cuda_stream()>>>(
        op, reinterpret_cast<uint8_t*>(output), bytes, dtype_info(dtype).bits, a, b, c, p);
}
} // namespace elman
