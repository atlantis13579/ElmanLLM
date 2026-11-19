#pragma once
#include "Core/DType.h"
#include <cstdint>

#ifdef __CUDACC__
#define ELMAN_PACKED_HD __host__ __device__
#else
#define ELMAN_PACKED_HD
#endif

namespace elman {
ELMAN_PACKED_HD inline constexpr bool packed_integer_signed(DType dtype) {
    return dtype == DType::Int1 || dtype == DType::Int2 || dtype == DType::Int3 ||
           dtype == DType::Int4 || dtype == DType::Int5 || dtype == DType::Int6 || dtype == DType::Int7;
}
ELMAN_PACKED_HD inline constexpr DType packed_integer_opmath(DType dtype) {
    return packed_integer_signed(dtype) ? DType::Int8 : DType::UInt8;
}
ELMAN_PACKED_HD inline int packed_integer_load(const void* input, uint64_t index, DType dtype) {
    const unsigned bits = packed_integer_bits(dtype);
    const uint64_t start = index * bits;
    const unsigned shift = unsigned(start % 8);
    const auto* bytes = static_cast<const uint8_t*>(input);
    unsigned code = unsigned(bytes[start / 8]) >> shift;
    if (shift + bits > 8) code |= unsigned(bytes[start / 8 + 1]) << (8 - shift);
    code &= (1u << bits) - 1u;
    return int(code) - (packed_integer_signed(dtype) && (code & (1u << (bits - 1))) ? int(1u << bits) : 0);
}
template<class Read>
ELMAN_PACKED_HD inline uint8_t packed_integer_byte(uint64_t byte, uint64_t count, unsigned bits, Read read) {
    // Ownership is per output byte, including values crossing byte boundaries.
    // Numeric results clear unused tail bits; storage copies preserve them.
    uint8_t result = 0;
    for (unsigned bit = 0; bit < 8; ++bit) {
        const uint64_t position = byte * 8 + bit, index = position / bits;
        if (index >= count) break;
        result |= uint8_t(((uint64_t(read(index)) >> (position % bits)) & 1u) << bit);
    }
    return result;
}
} // namespace elman
#undef ELMAN_PACKED_HD
