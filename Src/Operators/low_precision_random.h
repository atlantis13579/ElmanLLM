#pragma once
#include "Core/Backend.h"

namespace elman {
enum class LowPrecisionRandom { None, Uniform, Normal, Dropout };
inline LowPrecisionRandom low_precision_random_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype())) && !is_low_precision_float(DType(p.input_dtype(KernelInput::A)))) return LowPrecisionRandom::None;
    if (name == "uniform") return LowPrecisionRandom::Uniform;
    if (name == "normal") return LowPrecisionRandom::Normal;
    if (name == "dropout") return LowPrecisionRandom::Dropout;
    return LowPrecisionRandom::None;
}
void validate_low_precision_random(LowPrecisionRandom op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_random_cpu(LowPrecisionRandom op, uint32_t* output, const uint32_t* input, const KernelParams& p);
void launch_low_precision_random_cuda(LowPrecisionRandom op, uint32_t* output, const uint32_t* input, const KernelParams& p);
} // namespace elman
