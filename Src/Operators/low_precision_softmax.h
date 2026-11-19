#pragma once
#include "Core/Backend.h"

namespace elman {
// Attribute 1: 0 encodes forward output, 1 retains FP32 probabilities, 2 computes a leaf gradient.
enum class LowPrecisionSoftmax { None, Softmax, LogSoftmax };
inline LowPrecisionSoftmax low_precision_softmax_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype())) && !is_low_precision_float(DType(p.input_dtype(KernelInput::A))))
        return LowPrecisionSoftmax::None;
    if (name == "softmax") return LowPrecisionSoftmax::Softmax;
    if (name == "log_softmax") return LowPrecisionSoftmax::LogSoftmax;
    return LowPrecisionSoftmax::None;
}
void validate_low_precision_softmax(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_softmax_cpu(LowPrecisionSoftmax op, uint32_t* output, const uint32_t* input, const uint32_t* upstream, const KernelParams& p);
void launch_low_precision_softmax_cuda(LowPrecisionSoftmax op, uint32_t* output, const uint32_t* input, const uint32_t* upstream, const KernelParams& p);
} // namespace elman
