#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace elman {
enum class DType {
#define ELMAN_DTYPE(symbol, name, kind, bits) symbol,
#include "Core/DTypes.inc"
#undef ELMAN_DTYPE
};
enum class DTypeKind { Boolean, SignedInteger, UnsignedInteger, Floating, Complex, Quantized };
struct DTypeInfo {
    DType dtype;
    const char* name;
    DTypeKind kind;
    unsigned bits;
};
const DTypeInfo& dtype_info(DType dtype);
const char* dtype_name(DType dtype);
DType parse_dtype(const std::string& name);
std::vector<DType> all_dtypes();
size_t dtype_storage_bytes(DType dtype, size_t logical_count);
#ifdef __CUDACC__
#define ELMAN_DTYPE_HD __host__ __device__
#else
#define ELMAN_DTYPE_HD
#endif
ELMAN_DTYPE_HD inline constexpr bool is_compute_float(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
}
ELMAN_DTYPE_HD inline constexpr bool is_complex(DType dtype) {
    return dtype == DType::Complex32 || dtype == DType::Complex64 || dtype == DType::Complex128;
}
ELMAN_DTYPE_HD inline constexpr DType complex_component_dtype(DType dtype) {
    return dtype == DType::Complex32 ? DType::Float16 : dtype == DType::Complex128 ? DType::Float64 : DType::Float32;
}
// Arithmetic families are wider than the legacy Float32 opmath kernels.
ELMAN_DTYPE_HD inline constexpr bool is_arithmetic_float(DType dtype) {
    return is_compute_float(dtype) || dtype == DType::Float64;
}
ELMAN_DTYPE_HD inline constexpr bool is_narrow_float(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16;
}
ELMAN_DTYPE_HD inline constexpr bool is_float8(DType dtype) {
    return dtype >= DType::Float8E4M3FN && dtype <= DType::Float8E8M0FNU;
}
ELMAN_DTYPE_HD inline constexpr bool is_low_precision_float(DType dtype) {
    return is_float8(dtype) || dtype == DType::Float4E2M1FN;
}
// Byte-addressable integer formats; packed low-bit formats have separate rules.
ELMAN_DTYPE_HD inline constexpr bool is_standard_integer(DType dtype) {
    return dtype == DType::Bool || dtype == DType::Int8 || dtype == DType::UInt8 ||
           dtype == DType::Int16 || dtype == DType::UInt16 || dtype == DType::Int32 ||
           dtype == DType::UInt32 || dtype == DType::Int64 || dtype == DType::UInt64;
}
ELMAN_DTYPE_HD inline constexpr unsigned packed_integer_bits(DType dtype) {
    switch (dtype) {
    case DType::Int1: case DType::UInt1: return 1;
    case DType::Int2: case DType::UInt2: return 2;
    case DType::Int3: case DType::UInt3: return 3;
    case DType::Int4: case DType::UInt4: return 4;
    case DType::Int5: case DType::UInt5: return 5;
    case DType::Int6: case DType::UInt6: return 6;
    case DType::Int7: case DType::UInt7: return 7;
    default: return 0;
    }
}
ELMAN_DTYPE_HD inline constexpr bool is_packed_integer(DType dtype) {
    return packed_integer_bits(dtype) != 0;
}
ELMAN_DTYPE_HD inline constexpr bool is_arithmetic_integer(DType dtype) {
    return is_standard_integer(dtype) || is_packed_integer(dtype);
}
ELMAN_DTYPE_HD inline constexpr bool kernel_integer(int dtype) {
    return dtype == int(DType::Int32) || dtype == int(DType::Bool);
}
#undef ELMAN_DTYPE_HD
} // namespace elman
