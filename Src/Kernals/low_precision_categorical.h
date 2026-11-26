#pragma once
#include "Core/Backend.h"

namespace elman {
inline bool low_precision_categorical_operation(const std::string& name, const KernelParams& p) {
    return name == "categorical" && (is_low_precision_float(DType(p.input_dtype(KernelInput::A))) ||
        is_low_precision_float(DType(p.input_dtype(KernelInput::B))) || is_low_precision_float(DType(p.input_dtype(KernelInput::C))));
}
void validate_low_precision_categorical(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& params);
void launch_low_precision_categorical_cpu(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& params);
void launch_low_precision_categorical_cuda(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& params);
} // namespace elman
