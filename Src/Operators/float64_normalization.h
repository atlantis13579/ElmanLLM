#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Normalization { None, Layer, Rms, Batch, Group, GroupInputGradient, GroupWeightGradient };
inline Float64Normalization float64_normalization_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Normalization::None;
    if (name == "layer_norm") return Float64Normalization::Layer;
    if (name == "rms_norm") return Float64Normalization::Rms;
    if (name == "batch_norm") return Float64Normalization::Batch;
    if (name == "group_norm") return Float64Normalization::Group;
    if (name == "group_norm_input_grad") return Float64Normalization::GroupInputGradient;
    if (name == "group_norm_weight_grad") return Float64Normalization::GroupWeightGradient;
    return Float64Normalization::None;
}
void validate_float64_normalization(Float64Normalization op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_normalization_cpu(Float64Normalization op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_float64_normalization_cuda(Float64Normalization op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
