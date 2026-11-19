#pragma once
#include "Core/Backend.h"

namespace elman {
enum class LowPrecisionLoss { None, MeanSquaredError, MeanSquaredErrorGradient, CrossEntropy, CrossEntropyGradient };
inline LowPrecisionLoss low_precision_loss_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype())) && !is_low_precision_float(DType(p.input_dtype(KernelInput::A)))) return LowPrecisionLoss::None;
    if (name == "mse_loss") return p.attribute(0) == 1 ? LowPrecisionLoss::MeanSquaredErrorGradient : LowPrecisionLoss::MeanSquaredError;
    if (name == "cross_entropy") return LowPrecisionLoss::CrossEntropy;
    if (name == "cross_entropy_grad") return LowPrecisionLoss::CrossEntropyGradient;
    return LowPrecisionLoss::None;
}
inline int low_loss_count(const KernelParams& p) {
    int count = 1;
    for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
    return count;
}
inline KernelParams low_loss_softmax_params(const KernelParams& p) {
    auto result = p;
    result.v[KernelParams::kOutputCountSlot] = low_loss_count(p);
    result.v[KernelParams::kOutputRankSlot] = 2;
    result.v[KernelParams::kOutputShapeSlot] = p.input_shape(KernelInput::A, 0);
    result.v[KernelParams::kOutputShapeSlot + 1] = p.input_shape(KernelInput::A, 1);
    result.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32);
    for (int i = 1; i < 4; ++i) result.v[KernelParams::input_slot(KernelInput(i))] = -1;
    result.v[KernelParams::kAttributeSlot] = 1;
    result.v[KernelParams::kAttributeSlot + 1] = 1;
    return result;
}
void validate_low_precision_loss(LowPrecisionLoss op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_loss_cpu(LowPrecisionLoss op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_loss_cuda(LowPrecisionLoss op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
