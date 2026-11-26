#pragma once

#include "Core/Backend.h"

namespace elman {
enum class ComplexPointwise {
    None, Cast, Add, Sub, Mul, Div, Pow, Conj, Real, Imag, Abs, Angle, Sgn,
    Sqrt, Rsqrt, Exp, Log, Sin, Cos, Tanh, Eq, Ne, LogicalAnd, LogicalOr,
    LogicalNot, Where, Construct
};
ComplexPointwise complex_pointwise_operation(const std::string& name, const KernelParams& params);
int complex_pointwise_arity(ComplexPointwise operation);
void validate_complex_pointwise(ComplexPointwise operation, const BufferPtr& output,
                                 const std::vector<BufferPtr>& inputs, const KernelParams& params);
void launch_complex_pointwise_cpu(ComplexPointwise operation, uint32_t* output, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& params);
void launch_complex_pointwise_cuda(ComplexPointwise operation, uint32_t* output, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c, const KernelParams& params);
} // namespace elman
