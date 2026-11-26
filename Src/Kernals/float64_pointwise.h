#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Pointwise {
    None, Cast, Add, Sub, Mul, Div, Pow, Maximum, Minimum, Eq, Ne, Gt, Ge, Lt, Le,
    LogicalAnd, LogicalOr, LogicalNot, Sqrt, Rsqrt, Exp, Log, Sin, Cos, Abs, Sign,
    Relu, Silu, Gelu, LeakyRelu, Where, Fma, SiluMul, Rotary
};
inline int float64_pointwise_arity(Float64Pointwise op) {
    return op == Float64Pointwise::Where || op == Float64Pointwise::Fma || op == Float64Pointwise::Rotary ? 3 :
        (op >= Float64Pointwise::Add && op <= Float64Pointwise::LogicalOr) || op == Float64Pointwise::SiluMul ? 2 : 1;
}
inline Float64Pointwise float64_pointwise_operation(const std::string& name, const KernelParams& p) {
    bool wide = p.output_dtype() == int(DType::Float64);
    for (int i = 0; i < 3; ++i) wide |= p.input_rank(KernelInput(i)) >= 0 && p.input_dtype(KernelInput(i)) == int(DType::Float64);
    if (!wide) return Float64Pointwise::None;
    if (name == "cast" && (is_low_precision_float(DType(p.output_dtype())) ||
        is_low_precision_float(DType(p.input_dtype(KernelInput::A))))) return Float64Pointwise::None;
    if (name == "cast" && (is_standard_integer(DType(p.output_dtype())) || is_standard_integer(DType(p.input_dtype(KernelInput::A)))))
        return Float64Pointwise::None;
    if ((name == "logical_and" || name == "logical_or") &&
        (is_standard_integer(DType(p.input_dtype(KernelInput::A))) || is_standard_integer(DType(p.input_dtype(KernelInput::B)))))
        return Float64Pointwise::None;
#define ELMAN_FLOAT64_OP(label, kind) if (name == label) return Float64Pointwise::kind
    ELMAN_FLOAT64_OP("cast", Cast);
    ELMAN_FLOAT64_OP("add", Add);
    ELMAN_FLOAT64_OP("sub", Sub);
    ELMAN_FLOAT64_OP("mul", Mul);
    ELMAN_FLOAT64_OP("div", Div);
    ELMAN_FLOAT64_OP("pow", Pow);
    ELMAN_FLOAT64_OP("maximum", Maximum);
    ELMAN_FLOAT64_OP("minimum", Minimum);
    ELMAN_FLOAT64_OP("eq", Eq);
    ELMAN_FLOAT64_OP("ne", Ne);
    ELMAN_FLOAT64_OP("gt", Gt);
    ELMAN_FLOAT64_OP("ge", Ge);
    ELMAN_FLOAT64_OP("lt", Lt);
    ELMAN_FLOAT64_OP("le", Le);
    ELMAN_FLOAT64_OP("logical_and", LogicalAnd);
    ELMAN_FLOAT64_OP("logical_or", LogicalOr);
    ELMAN_FLOAT64_OP("logical_not", LogicalNot);
    ELMAN_FLOAT64_OP("sqrt", Sqrt);
    ELMAN_FLOAT64_OP("rsqrt", Rsqrt);
    ELMAN_FLOAT64_OP("exp", Exp);
    ELMAN_FLOAT64_OP("log", Log);
    ELMAN_FLOAT64_OP("sin", Sin);
    ELMAN_FLOAT64_OP("cos", Cos);
    ELMAN_FLOAT64_OP("abs", Abs);
    ELMAN_FLOAT64_OP("sign", Sign);
    ELMAN_FLOAT64_OP("relu", Relu);
    ELMAN_FLOAT64_OP("silu", Silu);
    ELMAN_FLOAT64_OP("swish", Silu);
    ELMAN_FLOAT64_OP("gelu", Gelu);
    ELMAN_FLOAT64_OP("leaky_relu", LeakyRelu);
    ELMAN_FLOAT64_OP("where", Where);
    ELMAN_FLOAT64_OP("float64_fma", Fma);
    ELMAN_FLOAT64_OP("silu_mul", SiluMul);
    ELMAN_FLOAT64_OP("rotary_embedding", Rotary);
#undef ELMAN_FLOAT64_OP
    return Float64Pointwise::None;
}
void validate_float64_pointwise(Float64Pointwise op, const BufferPtr& output,
                                const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_pointwise_cpu(Float64Pointwise op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_float64_pointwise_cuda(Float64Pointwise op, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
