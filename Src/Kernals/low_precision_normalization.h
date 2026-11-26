#pragma once
#include "Core/Backend.h"
namespace elman {
enum class LowPrecisionNormalization { None, Layer, Rms, Batch, Group, GroupInputGradient, GroupWeightGradient };
// Attribute 8 selects forward (0), input (1), weight (2), or bias (3) gradient.
// Attribute 9 selects training BatchNorm; its default raw contract uses supplied mean/variance.
#ifdef __CUDACC__
#define ELMAN_LOW_NORM_PARAMETER __host__ __device__
#else
#define ELMAN_LOW_NORM_PARAMETER
#endif
ELMAN_LOW_NORM_PARAMETER inline LowPrecisionNormalization low_precision_normalization_base(LowPrecisionNormalization op) {
    return op == LowPrecisionNormalization::GroupInputGradient || op == LowPrecisionNormalization::GroupWeightGradient ? LowPrecisionNormalization::Group : op;
}
ELMAN_LOW_NORM_PARAMETER inline int low_precision_normalization_mode(LowPrecisionNormalization op, const KernelParams& p) {
    return op == LowPrecisionNormalization::GroupInputGradient ? 1 : op == LowPrecisionNormalization::GroupWeightGradient ? 2 : p.attribute(8);
}
ELMAN_LOW_NORM_PARAMETER inline double low_precision_normalization_epsilon(LowPrecisionNormalization op, const KernelParams& p) {
    op = low_precision_normalization_base(op);
    if (p.attribute(7) || (op == LowPrecisionNormalization::Group && p.attribute(2))) {
        const int slot = p.attribute(7) ? 5 : 3;
        union { uint64_t bits; double value; } scalar;
        scalar.bits = uint64_t(uint32_t(p.attribute(slot))) | (uint64_t(uint32_t(p.attribute(slot + 1))) << 32);
        return scalar.value;
    }
    union { int bits; float value; } scalar;
    scalar.bits = p.attribute(op == LowPrecisionNormalization::Batch ? 0 : 1);
    return double(scalar.value);
}
inline LowPrecisionNormalization low_precision_normalization_operation(const std::string& name, const KernelParams& p) {
    // Private CPU Float32 LayerNorm shares the native statistics/derivative path.
    if (name == "layer_norm" && p.attribute(10) == 1 && p.output_dtype() == int(DType::Float32) &&
        p.input_dtype(KernelInput::A) == int(DType::Float32)) return LowPrecisionNormalization::Layer;
    if (!is_low_precision_float(DType(p.output_dtype())) && !is_low_precision_float(DType(p.input_dtype(KernelInput::A)))) return LowPrecisionNormalization::None;
    if (name == "layer_norm") return LowPrecisionNormalization::Layer;
    if (name == "rms_norm") return LowPrecisionNormalization::Rms;
    if (name == "batch_norm") return LowPrecisionNormalization::Batch;
    if (name == "group_norm") return LowPrecisionNormalization::Group;
    if (name == "group_norm_input_grad") return LowPrecisionNormalization::GroupInputGradient;
    if (name == "group_norm_weight_grad") return LowPrecisionNormalization::GroupWeightGradient;
    return LowPrecisionNormalization::None;
}
void validate_low_precision_normalization(LowPrecisionNormalization op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_normalization_cpu(LowPrecisionNormalization op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_normalization_cuda(LowPrecisionNormalization op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
