#pragma once
#include "Core/Backend.h"

namespace elman {
enum class LowPrecisionIndexing { None, GatherGradient, SelectionGradient, Scatter, ScatterAdd, ScatterGradient, SortGradient, RepeatGradient };
inline LowPrecisionIndexing low_precision_indexing_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype())))
        return LowPrecisionIndexing::None;
    if (name == "gather_grad" || name == "gather_backward") return LowPrecisionIndexing::GatherGradient;
    if (name == "index_select_grad" || name == "index_select_backward") return LowPrecisionIndexing::SelectionGradient;
    if (name == "scatter") return LowPrecisionIndexing::Scatter;
    if (name == "scatter_add") return LowPrecisionIndexing::ScatterAdd;
    if (name == "scatter_source_grad") return LowPrecisionIndexing::ScatterGradient;
    if (name == "sort_backward") return LowPrecisionIndexing::SortGradient;
    if (name == "repeat_grad" || name == "repeat_backward") return LowPrecisionIndexing::RepeatGradient;
    return LowPrecisionIndexing::None;
}
void validate_low_precision_indexing(LowPrecisionIndexing op, const BufferPtr& output,
                               const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_indexing_cpu(LowPrecisionIndexing op, uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_indexing_cuda(LowPrecisionIndexing op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
