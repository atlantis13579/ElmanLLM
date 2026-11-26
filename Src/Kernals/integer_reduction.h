#pragma once
#include "Core/DType.h"
#include <cstdint>

#ifdef __CUDACC__
#define ELMAN_INTEGER_HD __host__ __device__
#else
#define ELMAN_INTEGER_HD
#endif

namespace elman {
// Preserve integer bits throughout promotion and modulo-2^64 accumulation.
// Signed-to-unsigned conversion is defined even for the minimum signed value.
ELMAN_INTEGER_HD inline uint64_t integer_reduction_value(const void* source, int index, int dtype) {
    switch (DType(dtype)) {
    case DType::Bool: return reinterpret_cast<const uint8_t*>(source)[index] != 0;
    case DType::Int8: return uint64_t(int64_t(reinterpret_cast<const int8_t*>(source)[index]));
    case DType::UInt8: return reinterpret_cast<const uint8_t*>(source)[index];
    case DType::Int16: return uint64_t(int64_t(reinterpret_cast<const int16_t*>(source)[index]));
    case DType::UInt16: return reinterpret_cast<const uint16_t*>(source)[index];
    case DType::Int32: return uint64_t(int64_t(reinterpret_cast<const int32_t*>(source)[index]));
    case DType::UInt32: return reinterpret_cast<const uint32_t*>(source)[index];
    case DType::Int64:
    case DType::UInt64: return reinterpret_cast<const uint64_t*>(source)[index];
    default: return 0; // Common dispatch rejects all other input descriptors.
    }
}
} // namespace elman

#undef ELMAN_INTEGER_HD
