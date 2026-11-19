#pragma once
#include "Core/Tensor.h"

namespace elman {
enum class LowPrecisionAttention { None, Softmax, Tiled, Probabilities, Gradient, Sum, Matrix, SumTo };
inline LowPrecisionAttention low_precision_attention_operation(const std::string& name, const KernelParams& p) {
    if (name == "attention_softmax" && p.output_dtype() == int(DType::Float32)) {
        if (p.attribute(8) == 1) return LowPrecisionAttention::Gradient;
        if (p.attribute(8) == 2) return LowPrecisionAttention::Probabilities;
        if (p.attribute(8) == 3) return LowPrecisionAttention::Sum;
        if (p.attribute(8) == 4) return LowPrecisionAttention::Matrix;
        if (p.attribute(8) == 5) return LowPrecisionAttention::SumTo;
    }
    if (!is_low_precision_float(DType(p.output_dtype()))) return LowPrecisionAttention::None;
    if (name == "attention_softmax") return LowPrecisionAttention::Softmax;
    if (name == "tiled_attention") return LowPrecisionAttention::Tiled;
    return LowPrecisionAttention::None;
}
void validate_low_precision_attention(LowPrecisionAttention op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void execute_low_precision_attention(LowPrecisionAttention op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_attention_sum_cpu(uint32_t* output, const uint32_t* input, const KernelParams& p);
void launch_low_attention_sum_cuda(uint32_t* output, const uint32_t* input, const KernelParams& p);
void launch_low_attention_matrix_cpu(uint32_t* output, const uint32_t* a, const uint32_t* b, const KernelParams& p);
void launch_low_attention_matrix_cuda(uint32_t* output, const uint32_t* a, const uint32_t* b, const KernelParams& p);
Tensor low_attention_reduce_shape(const Tensor& gradient, const Shape& shape);
Tensor low_attention_softmax_derivative(const Tensor& scores, const Tensor& mask, double divisor, int axis, const Tensor& gradient);
std::vector<Tensor> low_tiled_attention_derivatives(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& mask, double divisor, bool causal, int offset, const Tensor& gradient);
inline KernelParams low_attention_softmax_parameters(const KernelParams& p, LowPrecisionAttention op) {
    auto result = p; result.v[KernelParams::kAttributeSlot + 1] = op == LowPrecisionAttention::Gradient ? 2 : 1; return result;
}
} // namespace elman
