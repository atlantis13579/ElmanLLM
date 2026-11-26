#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Random { None, Uniform, Normal, Dropout };
inline Float64Random float64_random_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Random::None;
    if (name == "uniform") return Float64Random::Uniform;
    if (name == "normal") return Float64Random::Normal;
    if (name == "dropout") return Float64Random::Dropout;
    return Float64Random::None;
}
void validate_float64_random(Float64Random op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_random_cpu(Float64Random op, uint32_t* output, const uint32_t* input, const KernelParams& p);
void launch_float64_random_cuda(Float64Random op, uint32_t* output, const uint32_t* input, const KernelParams& p);
} // namespace elman
