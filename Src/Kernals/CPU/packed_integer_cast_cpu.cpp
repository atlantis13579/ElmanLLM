#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Kernals/packed_integer_cast.h"
#include "Kernals/integer_pointwise.h"
#include <cstring>

namespace elman {
void launch_packed_integer_cast_cpu(uint32_t* output, const uint32_t* input, const KernelParams& p) {
    const DType source = DType(p.input_dtype(KernelInput::A)), target = DType(p.output_dtype());
    const uint64_t count = uint64_t(p.output_count());
    if (!count) return;
    if (source == target) {
        if (output != input) std::memcpy(output, input, dtype_storage_bytes(target, size_t(count)));
        return;
    }
    LocalFunction read{[&](uint64_t index) {
        return is_packed_integer(source) ? uint64_t(int64_t(packed_integer_load(input, index, source))) :
                                          integer_reduction_value(input, int(index), int(source));
    }};
    if (const unsigned bits = packed_integer_bits(target)) {
        uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
        for (uint64_t byte = 0; byte < (count * bits + 7) / 8; ++byte)
            bytes[byte] = packed_integer_byte(byte, count, bits, read);
    } else for (uint64_t i = 0; i < count; ++i) integer_write(output, int(i), int(target), read(i));
}
} // namespace elman
