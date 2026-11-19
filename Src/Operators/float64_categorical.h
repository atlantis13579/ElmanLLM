#pragma once
#include "Core/Backend.h"

namespace elman {
inline bool float64_categorical_operation(const std::string& name, const KernelParams& p) {
    return name == "categorical" && (p.input_dtype(KernelInput::A) == int(DType::Float64) ||
        p.input_dtype(KernelInput::B) == int(DType::Float64) || p.input_dtype(KernelInput::C) == int(DType::Float64));
}
void validate_float64_categorical(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_categorical_cpu(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& p);
void launch_float64_categorical_cuda(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& p);
} // namespace elman
