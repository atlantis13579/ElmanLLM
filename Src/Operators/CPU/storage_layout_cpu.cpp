#include "Operators/storage_layout.h"

namespace elman {
void launch_storage_layout_cpu(StorageLayout op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    const unsigned bits = dtype_info(dtype).bits;
    const size_t bytes = dtype_storage_bytes(dtype, size_t(p.output_count()));
    uint8_t* destination = reinterpret_cast<uint8_t*>(output);
    for (size_t i = 0; i < bytes; ++i) destination[i] = storage_layout_byte(op, i, bits, a, b, c, p);
}
} // namespace elman
