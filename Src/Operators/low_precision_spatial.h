#pragma once
#include "Core/Backend.h"

namespace elman {
enum class LowPrecisionSpatial { None, Convolution, InputGradient, WeightGradient, MaximumPool, AveragePool, MaximumPoolGradient, AveragePoolGradient };
inline LowPrecisionSpatial low_precision_spatial_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype())))
        return LowPrecisionSpatial::None;
    if (name == "conv2d") return LowPrecisionSpatial::Convolution;
    if (name == "conv2d_input_grad") return LowPrecisionSpatial::InputGradient;
    if (name == "conv2d_weight_grad") return LowPrecisionSpatial::WeightGradient;
    if (name == "max_pool2d") return LowPrecisionSpatial::MaximumPool;
    if (name == "avg_pool2d") return LowPrecisionSpatial::AveragePool;
    if (name == "max_pool2d_grad") return LowPrecisionSpatial::MaximumPoolGradient;
    if (name == "avg_pool2d_grad") return LowPrecisionSpatial::AveragePoolGradient;
    return LowPrecisionSpatial::None;
}
void validate_low_precision_spatial(LowPrecisionSpatial op, const BufferPtr& output,
                              const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_spatial_cpu(LowPrecisionSpatial op, uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_spatial_cuda(LowPrecisionSpatial op, uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
