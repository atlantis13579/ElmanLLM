#pragma once

#include "Core/LocalFunction.h"
#include "Core/Backend.h"
#include "Operators/integer_reduction.h"

#ifdef __CUDACC__
#define ELMAN_INTEGER_HD __host__ __device__
#else
#define ELMAN_INTEGER_HD
#endif

namespace elman {
enum class IntegerPointwise {
    None, Cast, Add, Sub, Mul, Pow, Maximum, Minimum, Eq, Ne, Gt, Ge, Lt, Le,
    LogicalAnd, LogicalOr, LogicalNot, Abs, Sign, Relu, Where
};
ELMAN_INTEGER_HD inline int integer_pointwise_arity(IntegerPointwise op) {
    return op == IntegerPointwise::Where ? 3 : op == IntegerPointwise::Cast ||
        op == IntegerPointwise::LogicalNot || op == IntegerPointwise::Abs ||
        op == IntegerPointwise::Sign || op == IntegerPointwise::Relu ? 1 : 2;
}

inline IntegerPointwise integer_pointwise_operation(const std::string& name, const KernelParams& p) {
    const DType a = DType(p.input_dtype(KernelInput::A));
    const DType b = DType(p.input_dtype(KernelInput::B));
    const DType out = DType(p.output_dtype());
    if (name == "cast" && (is_standard_integer(a) || is_standard_integer(out))) return IntegerPointwise::Cast;
    if (name == "where") {
        if (kernel_integer(int(b)) && (p.input_dtype(KernelInput::C) != int(b) || out != b))
            return IntegerPointwise::None;
        return is_standard_integer(b) ? IntegerPointwise::Where : IntegerPointwise::None;
    }
    if (name == "logical_and" || name == "logical_or") {
        if (!is_standard_integer(a) && !is_standard_integer(b)) return IntegerPointwise::None;
        return name == "logical_and" ? IntegerPointwise::LogicalAnd : IntegerPointwise::LogicalOr;
    }
    if (!is_standard_integer(a)) return IntegerPointwise::None;
    LocalFunction select{[&](IntegerPointwise kind) {
        // Preserve the established mixed Int32/Bool/float raw-kernel ABI.
        // Public integer operations promote operands before dispatch.
        if (integer_pointwise_arity(kind) == 2 && a != b && kernel_integer(int(a)) &&
            (kernel_integer(int(b)) || is_compute_float(b))) return IntegerPointwise::None;
        return kind;
    }};
#define ELMAN_INTEGER_OP(label, kind) if (operation_name == label) return select(IntegerPointwise::kind)
    const std::string& operation_name = name;
    ELMAN_INTEGER_OP("add", Add);
    ELMAN_INTEGER_OP("sub", Sub);
    ELMAN_INTEGER_OP("mul", Mul);
    ELMAN_INTEGER_OP("pow", Pow);
    ELMAN_INTEGER_OP("maximum", Maximum);
    ELMAN_INTEGER_OP("minimum", Minimum);
    ELMAN_INTEGER_OP("eq", Eq);
    ELMAN_INTEGER_OP("ne", Ne);
    ELMAN_INTEGER_OP("gt", Gt);
    ELMAN_INTEGER_OP("ge", Ge);
    ELMAN_INTEGER_OP("lt", Lt);
    ELMAN_INTEGER_OP("le", Le);
    ELMAN_INTEGER_OP("logical_not", LogicalNot);
    ELMAN_INTEGER_OP("abs", Abs);
    ELMAN_INTEGER_OP("sign", Sign);
    ELMAN_INTEGER_OP("relu", Relu);
#undef ELMAN_INTEGER_OP
    return IntegerPointwise::None;
}

ELMAN_INTEGER_HD inline bool integer_signed(int dtype) {
    return dtype == int(DType::Int8) || dtype == int(DType::Int16) ||
           dtype == int(DType::Int32) || dtype == int(DType::Int64);
}
ELMAN_INTEGER_HD inline bool integer_negative(uint64_t value, int dtype) {
    return integer_signed(dtype) && (value >> 63) != 0;
}
ELMAN_INTEGER_HD inline void integer_write(void* output, int index, int dtype, uint64_t value) {
    switch (DType(dtype)) {
    case DType::Bool: reinterpret_cast<uint8_t*>(output)[index] = value != 0; break;
    case DType::Int8:
    case DType::UInt8: reinterpret_cast<uint8_t*>(output)[index] = uint8_t(value); break;
    case DType::Int16:
    case DType::UInt16: reinterpret_cast<uint16_t*>(output)[index] = uint16_t(value); break;
    case DType::Int32:
    case DType::UInt32: reinterpret_cast<uint32_t*>(output)[index] = uint32_t(value); break;
    case DType::Int64:
    case DType::UInt64: reinterpret_cast<uint64_t*>(output)[index] = value; break;
    default: break;
    }
}
template<class Float>
ELMAN_INTEGER_HD inline Float integer_to_float(uint64_t value, int dtype) {
    return integer_negative(value, dtype) ? -Float(uint64_t(0) - value) : Float(value);
}
ELMAN_INTEGER_HD inline uint64_t integer_pointwise_value(IntegerPointwise op, uint64_t x,
                                                        uint64_t y, uint64_t z, int dtype) {
    // Unsigned arithmetic defines wrapping for every integer width, including Int64.
    const uint64_t sign = integer_signed(dtype) ? uint64_t(1) << 63 : 0;
    const bool less = (x ^ sign) < (y ^ sign);
    switch (op) {
    case IntegerPointwise::Add: return x + y;
    case IntegerPointwise::Sub: return x - y;
    case IntegerPointwise::Mul: return x * y;
    case IntegerPointwise::Pow: {
        if (integer_negative(y, dtype)) return x == 1 ? 1 : x == UINT64_MAX ? (y & 1 ? UINT64_MAX : 1) : 0;
        uint64_t result = 1;
        for (; y; y >>= 1, x *= x) if (y & 1) result *= x;
        return result;
    }
    case IntegerPointwise::Maximum: return less ? y : x;
    case IntegerPointwise::Minimum: return less ? x : y;
    case IntegerPointwise::Eq: return x == y;
    case IntegerPointwise::Ne: return x != y;
    case IntegerPointwise::Gt: return !less && x != y;
    case IntegerPointwise::Ge: return !less;
    case IntegerPointwise::Lt: return less;
    case IntegerPointwise::Le: return less || x == y;
    case IntegerPointwise::LogicalAnd: return x && y;
    case IntegerPointwise::LogicalOr: return x || y;
    case IntegerPointwise::LogicalNot: return !x;
    case IntegerPointwise::Abs: return integer_negative(x, dtype) ? uint64_t(0) - x : x;
    case IntegerPointwise::Sign: return integer_negative(x, dtype) ? UINT64_MAX : uint64_t(x != 0);
    case IntegerPointwise::Relu: return integer_negative(x, dtype) ? 0 : x;
    case IntegerPointwise::Where: return x ? y : z;
    default: return x;
    }
}
void validate_integer_pointwise(IntegerPointwise op, const BufferPtr& output,
                                const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_integer_pointwise_cpu(IntegerPointwise op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_integer_pointwise_cuda(IntegerPointwise op, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
#undef ELMAN_INTEGER_HD
