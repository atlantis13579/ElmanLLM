#include "Core/DType.h"
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
constexpr DTypeInfo types[] = {
#define ELMAN_DTYPE(symbol, name, kind, bits) {DType::symbol, name, DTypeKind::kind, bits},
#include "Core/DTypes.inc"
#undef ELMAN_DTYPE
};
}
const DTypeInfo& dtype_info(DType dtype) {
    size_t index = static_cast<size_t>(dtype);
    if (index >= sizeof(types) / sizeof(types[0])) throw std::invalid_argument("Unknown tensor dtype");
    return types[index];
}
const char* dtype_name(DType dtype) { return dtype_info(dtype).name; }
DType parse_dtype(const std::string& name) {
    for (const DTypeInfo& type : types) if (name == type.name) return type.dtype;
    throw std::invalid_argument("Unknown tensor dtype: " + name);
}
std::vector<DType> all_dtypes() {
    std::vector<DType> result;
    for (const DTypeInfo& type : types) result.push_back(type.dtype);
    return result;
}
size_t dtype_storage_bytes(DType dtype, size_t logical_count) {
    unsigned int bits = dtype_info(dtype).bits;
    // Avoid overflowing count*bits for valid large packed allocations.
    size_t groups = logical_count / 8;
    size_t tail = ((logical_count % 8) * bits + 7) / 8;
    if (groups > (std::numeric_limits<size_t>::max() - tail) / bits)
        throw std::overflow_error("Tensor storage byte count overflow");
    return groups * bits + tail;
}
} // namespace elman
