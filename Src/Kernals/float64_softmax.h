#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Softmax { None, Softmax, LogSoftmax };
inline Float64Softmax float64_softmax_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Softmax::None;
    if (name == "softmax") return Float64Softmax::Softmax;
    if (name == "log_softmax") return Float64Softmax::LogSoftmax;
    return Float64Softmax::None;
}
void validate_float64_softmax(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_softmax_cpu(Float64Softmax op, uint32_t* output, const uint32_t* input, const KernelParams& p);
void launch_float64_softmax_cuda(Float64Softmax op, uint32_t* output, const uint32_t* input, const KernelParams& p);
} // namespace elman
