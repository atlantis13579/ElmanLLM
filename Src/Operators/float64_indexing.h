#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Indexing { None, GatherGradient, SelectionGradient, Scatter, ScatterAdd, ScatterGradient, SortGradient, RepeatGradient };
inline Float64Indexing float64_indexing_operation(const std::string& name, const KernelParams& p) {
    if (p.output_dtype() != int(DType::Float64) && p.input_dtype(KernelInput::A) != int(DType::Float64))
        return Float64Indexing::None;
    if (name == "gather_grad" || name == "gather_backward") return Float64Indexing::GatherGradient;
    if (name == "index_select_grad" || name == "index_select_backward") return Float64Indexing::SelectionGradient;
    if (name == "scatter") return Float64Indexing::Scatter;
    if (name == "scatter_add") return Float64Indexing::ScatterAdd;
    if (name == "scatter_source_grad") return Float64Indexing::ScatterGradient;
    if (name == "sort_backward") return Float64Indexing::SortGradient;
    if (name == "repeat_grad" || name == "repeat_backward") return Float64Indexing::RepeatGradient;
    return Float64Indexing::None;
}
void validate_float64_indexing(Float64Indexing op, const BufferPtr& output,
                               const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_indexing_cpu(Float64Indexing op, uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_float64_indexing_cuda(Float64Indexing op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
