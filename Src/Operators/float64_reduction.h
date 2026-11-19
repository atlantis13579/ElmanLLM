#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Reduction { None, Sum, Mean, Product, Maximum, Minimum, Gradient, Cumsum, ReverseCumsum };
inline Float64Reduction float64_reduction_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Reduction::None;
    if (name == "sum") return Float64Reduction::Sum;
    if (name == "mean") return Float64Reduction::Mean;
    if (name == "prod") return Float64Reduction::Product;
    if (name == "max" || name == "amax") return Float64Reduction::Maximum;
    if (name == "min" || name == "amin") return Float64Reduction::Minimum;
    if (name == "reduce_grad") return Float64Reduction::Gradient;
    if (name == "cumsum") return Float64Reduction::Cumsum;
    if (name == "cumsum_backward") return Float64Reduction::ReverseCumsum;
    return Float64Reduction::None;
}
void validate_float64_reduction(Float64Reduction op, const BufferPtr& output,
                                const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_reduction_cpu(Float64Reduction op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_float64_reduction_cuda(Float64Reduction op, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
