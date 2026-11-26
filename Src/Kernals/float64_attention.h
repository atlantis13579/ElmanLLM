#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Attention { None, Softmax, Tiled };
inline Float64Attention float64_attention_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Attention::None;
    if (name == "attention_softmax") return Float64Attention::Softmax;
    if (name == "tiled_attention") return Float64Attention::Tiled;
    return Float64Attention::None;
}
void validate_float64_attention(Float64Attention op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_attention_cpu(Float64Attention op, uint32_t* output, const uint32_t* a,
    const uint32_t* b, const uint32_t* c, const uint32_t* d, const KernelParams& p);
void launch_float64_attention_cuda(Float64Attention op, uint32_t* output, const uint32_t* a,
    const uint32_t* b, const uint32_t* c, const uint32_t* d, const KernelParams& p);
} // namespace elman
