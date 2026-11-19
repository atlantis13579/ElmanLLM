#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Loss { None, MeanSquaredError, CrossEntropy, CrossEntropyGradient };
inline Float64Loss float64_loss_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Loss::None;
    if (name == "mse_loss") return Float64Loss::MeanSquaredError;
    if (name == "cross_entropy") return Float64Loss::CrossEntropy;
    if (name == "cross_entropy_grad") return Float64Loss::CrossEntropyGradient;
    return Float64Loss::None;
}
void validate_float64_loss(Float64Loss op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_loss_cpu(Float64Loss op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_float64_loss_cuda(Float64Loss op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
