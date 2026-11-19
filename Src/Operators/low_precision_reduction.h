#pragma once
#include "Core/Backend.h"

namespace elman {
enum class LowPrecisionReduction { None, Sum, Mean, Product, Maximum, Minimum, Amax, Amin, Gradient, Cumsum, ReverseCumsum };
inline LowPrecisionReduction low_precision_reduction_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype())) && !is_low_precision_float(DType(p.input_dtype(KernelInput::A))))
        return LowPrecisionReduction::None;
    if (name == "sum") return LowPrecisionReduction::Sum;
    if (name == "mean") return LowPrecisionReduction::Mean;
    if (name == "prod") return LowPrecisionReduction::Product;
    if (name == "amax") return LowPrecisionReduction::Amax;
    if (name == "amin") return LowPrecisionReduction::Amin;
    if (name == "max") return LowPrecisionReduction::Maximum;
    if (name == "min") return LowPrecisionReduction::Minimum;
    if (name == "reduce_grad") return LowPrecisionReduction::Gradient;
    if (name == "cumsum") return LowPrecisionReduction::Cumsum;
    if (name == "cumsum_backward") return LowPrecisionReduction::ReverseCumsum;
    return LowPrecisionReduction::None;
}
void validate_low_precision_reduction(LowPrecisionReduction op, const BufferPtr& output,
                                const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_reduction_cpu(LowPrecisionReduction op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_reduction_cuda(LowPrecisionReduction op, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
