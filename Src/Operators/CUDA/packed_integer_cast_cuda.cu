#include "kernel.cuh"
#include "Operators/packed_integer_cast.h"
#include "Operators/integer_pointwise.h"

namespace elman {
namespace {
struct PackedReader {
    const uint32_t* input;
    DType dtype;
    __host__ __device__ uint64_t operator()(uint64_t index) const {
        return is_packed_integer(dtype) ? uint64_t(int64_t(packed_integer_load(input, index, dtype))) :
                                         integer_reduction_value(input, int(index), int(dtype));
    }
};
__global__ void packed_integer_cast_kernel(uint32_t* output, const uint32_t* input, int count, DType source, DType target) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned bits = packed_integer_bits(target);
    if (bits) {
        if (index >= (uint64_t(count) * bits + 7) / 8) return;
        reinterpret_cast<uint8_t*>(output)[index] = source == target ? reinterpret_cast<const uint8_t*>(input)[index] :
            packed_integer_byte(index, uint64_t(count), bits, PackedReader{input, source});
    } else if (index < uint64_t(count)) {
        integer_write(output, int(index), int(target), PackedReader{input, source}(index));
    }
}
}
void launch_packed_integer_cast_cuda(uint32_t* output, const uint32_t* input, const KernelParams& p) {
    if (!p.output_count()) return;
    const auto source = DType(p.input_dtype(KernelInput::A)), target = DType(p.output_dtype());
    const unsigned bits = packed_integer_bits(target);
    const uint64_t work = bits ? (uint64_t(p.output_count()) * bits + 7) / 8 : uint64_t(p.output_count());
    packed_integer_cast_kernel<<<unsigned((work + 255) / 256), 256, 0, current_cuda_stream()>>>(output, input, p.output_count(), source, target);
}
} // namespace elman
