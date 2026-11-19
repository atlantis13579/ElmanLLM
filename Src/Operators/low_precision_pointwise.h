#pragma once
#include "Core/Backend.h"

namespace elman {
enum class LowPrecisionPointwise {
    None, Add, Sub, Mul, Div, Pow, Maximum, Minimum, Eq, Ne, Gt, Ge, Lt, Le,
    LogicalAnd, LogicalOr, LogicalNot, Sqrt, Rsqrt, Exp, Log, Sin, Cos, Abs, Sign,
    Relu, Silu, Gelu, LeakyRelu, Where
};
#ifdef __CUDACC__
__host__ __device__
#endif
inline int low_precision_pointwise_arity(LowPrecisionPointwise op) {
    return op == LowPrecisionPointwise::Where ? 3 :
        op >= LowPrecisionPointwise::Add && op <= LowPrecisionPointwise::LogicalOr ? 2 : 1;
}
inline bool low_precision_pointwise_boolean(LowPrecisionPointwise op) {
    return op >= LowPrecisionPointwise::Eq && op <= LowPrecisionPointwise::LogicalNot;
}
inline LowPrecisionPointwise low_precision_pointwise_operation(const std::string& name, const KernelParams& p) {
    bool low = is_low_precision_float(DType(p.output_dtype()));
    for (int i = 0; i < 3; ++i)
        low |= p.input_rank(KernelInput(i)) >= 0 && is_low_precision_float(DType(p.input_dtype(KernelInput(i))));
    if (!low) return LowPrecisionPointwise::None;
#define ELMAN_LOW_PRECISION_OP(label, kind) if (name == label) return LowPrecisionPointwise::kind
    ELMAN_LOW_PRECISION_OP("add", Add);
    ELMAN_LOW_PRECISION_OP("sub", Sub);
    ELMAN_LOW_PRECISION_OP("mul", Mul);
    ELMAN_LOW_PRECISION_OP("div", Div);
    ELMAN_LOW_PRECISION_OP("pow", Pow);
    ELMAN_LOW_PRECISION_OP("maximum", Maximum);
    ELMAN_LOW_PRECISION_OP("minimum", Minimum);
    ELMAN_LOW_PRECISION_OP("eq", Eq);
    ELMAN_LOW_PRECISION_OP("ne", Ne);
    ELMAN_LOW_PRECISION_OP("gt", Gt);
    ELMAN_LOW_PRECISION_OP("ge", Ge);
    ELMAN_LOW_PRECISION_OP("lt", Lt);
    ELMAN_LOW_PRECISION_OP("le", Le);
    ELMAN_LOW_PRECISION_OP("logical_and", LogicalAnd);
    ELMAN_LOW_PRECISION_OP("logical_or", LogicalOr);
    ELMAN_LOW_PRECISION_OP("logical_not", LogicalNot);
    ELMAN_LOW_PRECISION_OP("sqrt", Sqrt);
    ELMAN_LOW_PRECISION_OP("rsqrt", Rsqrt);
    ELMAN_LOW_PRECISION_OP("exp", Exp);
    ELMAN_LOW_PRECISION_OP("log", Log);
    ELMAN_LOW_PRECISION_OP("sin", Sin);
    ELMAN_LOW_PRECISION_OP("cos", Cos);
    ELMAN_LOW_PRECISION_OP("abs", Abs);
    ELMAN_LOW_PRECISION_OP("sign", Sign);
    ELMAN_LOW_PRECISION_OP("relu", Relu);
    ELMAN_LOW_PRECISION_OP("silu", Silu);
    ELMAN_LOW_PRECISION_OP("swish", Silu);
    ELMAN_LOW_PRECISION_OP("gelu", Gelu);
    ELMAN_LOW_PRECISION_OP("leaky_relu", LeakyRelu);
    ELMAN_LOW_PRECISION_OP("where", Where);
#undef ELMAN_LOW_PRECISION_OP
    return LowPrecisionPointwise::None;
}
void validate_low_precision_pointwise(LowPrecisionPointwise op, const BufferPtr& output,
                                      const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_pointwise_cpu(LowPrecisionPointwise op, uint32_t* output, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_pointwise_cuda(LowPrecisionPointwise op, uint32_t* output, const uint32_t* a,
                                         const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
