#pragma once
#include "Core/Backend.h"

namespace elman {
enum class LowPrecisionFused { None, SiluMul, Rotary };
inline LowPrecisionFused low_precision_fused_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype()))) return LowPrecisionFused::None;
    if (name == "silu_mul") return LowPrecisionFused::SiluMul;
    if (name == "rotary_embedding") return LowPrecisionFused::Rotary;
    return LowPrecisionFused::None;
}
void validate_low_precision_fused(LowPrecisionFused op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_fused_cpu(LowPrecisionFused op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_fused_cuda(LowPrecisionFused op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
