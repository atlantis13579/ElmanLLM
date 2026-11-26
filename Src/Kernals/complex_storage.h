#pragma once

#include "Kernals/complex_math.h"
#ifdef __CUDACC__
#define ELMAN_COMPLEX_STORAGE __device__
#else
#define ELMAN_COMPLEX_STORAGE
#endif
namespace elman {
template<class Real, class FloatReader>
ELMAN_COMPLEX_STORAGE ComplexNumber<Real> complex_read(const uint32_t* data, uint64_t index,
                                                       DType dtype, FloatReader floating) {
    if (is_complex(dtype)) {
        const DType component = complex_component_dtype(dtype);
        return {Real(floating(data, index * 2, component)), Real(floating(data, index * 2 + 1, component))};
    }
    if (is_arithmetic_float(dtype)) return {Real(floating(data, index, dtype)), 0};
    switch (dtype) {
    case DType::Bool: return {Real(reinterpret_cast<const uint8_t*>(data)[index] != 0), 0};
    case DType::Int8: return {Real(reinterpret_cast<const int8_t*>(data)[index]), 0};
    case DType::UInt8: return {Real(reinterpret_cast<const uint8_t*>(data)[index]), 0};
    case DType::Int16: return {Real(reinterpret_cast<const int16_t*>(data)[index]), 0};
    case DType::UInt16: return {Real(reinterpret_cast<const uint16_t*>(data)[index]), 0};
    case DType::Int32: return {Real(reinterpret_cast<const int32_t*>(data)[index]), 0};
    case DType::UInt32: return {Real(reinterpret_cast<const uint32_t*>(data)[index]), 0};
    case DType::Int64: return {Real(reinterpret_cast<const int64_t*>(data)[index]), 0};
    case DType::UInt64: return {Real(reinterpret_cast<const uint64_t*>(data)[index]), 0};
    default: return {};
    }
}
template<class Real, class FloatWriter>
ELMAN_COMPLEX_STORAGE void complex_write(uint32_t* data, uint64_t index, DType dtype,
                                          ComplexNumber<Real> value, FloatWriter floating) {
    if (is_complex(dtype)) {
        const DType component = complex_component_dtype(dtype);
        floating(data, index * 2, component, value.real());
        floating(data, index * 2 + 1, component, value.imag());
    } else if (dtype == DType::Bool) {
        reinterpret_cast<uint8_t*>(data)[index] = value.real() != 0 || value.imag() != 0;
    } else floating(data, index, dtype, value.real());
}
inline bool complex_pointwise_wide(const KernelParams& p) {
    if (p.output_dtype() == int(DType::Complex128) || p.output_dtype() == int(DType::Float64)) return true;
    for (KernelInput input : {KernelInput::A, KernelInput::B, KernelInput::C})
        if (p.input_rank(input) >= 0 && (p.input_dtype(input) == int(DType::Complex128) || p.input_dtype(input) == int(DType::Float64))) return true;
    return false;
}
} // namespace elman
#undef ELMAN_COMPLEX_STORAGE
