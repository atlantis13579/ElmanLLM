#pragma once
#include "Core/Backend.h"
#include "Kernals/attributes.h"
#include <type_traits>

namespace elman {
inline KernelParams projection_params(const KernelParams& packed, int width) {
    KernelParams result = packed;
    int rank = packed.input_rank(KernelInput::A), rows = 1;
    result.v[KernelParams::kOutputRankSlot] = rank;
    for (int axis = 0; axis + 1 < rank; ++axis) {
        int size = packed.input_shape(KernelInput::A, axis);
        rows *= size;
        result.v[KernelParams::kOutputShapeSlot + axis] = size;
    }
    result.v[KernelParams::kOutputShapeSlot + rank - 1] = width;
    result.v[KernelParams::kOutputCountSlot] = rows * width;
    result.v[KernelParams::input_slot(KernelInput::B) + KernelParams::kInputShapeOffset] = width;
    if (packed.input_rank(KernelInput::C) >= 0)
        result.v[KernelParams::input_slot(KernelInput::C) + KernelParams::kInputShapeOffset] = width;
    for (int i = 0; i < KernelParams::kAttributeCount; ++i) result.v[KernelParams::kAttributeSlot + i] = 0;
    return result;
}
template<class T> T* projection_offset(T* data, size_t elements, size_t bytes) {
    using Byte = typename std::conditional<std::is_const<T>::value, const uint8_t, uint8_t>::type;
    return elements ? reinterpret_cast<T*>(reinterpret_cast<Byte*>(data) + elements * bytes) : data;
}
} // namespace elman
