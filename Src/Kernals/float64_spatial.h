#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Spatial { None, Convolution, InputGradient, WeightGradient, MaximumPool, AveragePool, MaximumPoolGradient, AveragePoolGradient };
inline Float64Spatial float64_spatial_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Spatial::None;
    if (name == "conv2d") return Float64Spatial::Convolution;
    if (name == "conv2d_input_grad") return Float64Spatial::InputGradient;
    if (name == "conv2d_weight_grad") return Float64Spatial::WeightGradient;
    if (name == "max_pool2d") return Float64Spatial::MaximumPool;
    if (name == "avg_pool2d") return Float64Spatial::AveragePool;
    if (name == "max_pool2d_grad") return Float64Spatial::MaximumPoolGradient;
    if (name == "avg_pool2d_grad") return Float64Spatial::AveragePoolGradient;
    return Float64Spatial::None;
}
void validate_float64_spatial(Float64Spatial op, const BufferPtr& output,
                              const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_spatial_cpu(Float64Spatial op, uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_float64_spatial_cuda(Float64Spatial op, uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
