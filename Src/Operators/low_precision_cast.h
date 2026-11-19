#pragma once
#include "Core/Backend.h"

namespace elman {
inline bool low_precision_cast_operation(const std::string& name, const KernelParams& p) {
    return name == "cast" && (is_low_precision_float(DType(p.output_dtype())) ||
        is_low_precision_float(DType(p.input_dtype(KernelInput::A))));
}
void validate_low_precision_cast(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_cast_cpu(uint32_t* output, const uint32_t* input, const KernelParams& p);
void launch_low_precision_cast_cuda(uint32_t* output, const uint32_t* input, const KernelParams& p);
} // namespace elman
