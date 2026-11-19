#include "Inference/LLM/ProjectionWeight.h"
#include <stdexcept>

namespace elman {
ProjectionWeight ProjectionWeight::to_device(int ordinal) const {
    if (!affine.defined()) return ProjectionWeight(dense.to_device(ordinal));
    return ProjectionWeight(QuantizedTensor(affine.codes().to_device(ordinal), affine.scales().to_device(ordinal),
                                           affine.zero_points().to_device(ordinal), affine.group_size()));
}
Tensor ProjectionWeight::materialize(const Tensor& workspace) const {
    if (!affine.defined()) return dense;
    auto result = contiguous_view(workspace, affine.shape());
    affine.dequantize_into(result);
    return result;
}
Tensor ProjectionWeight::linear(const Tensor& input, const Tensor& bias, const Tensor& workspace) const {
    if (!affine.defined()) return elman::linear(input, dense, bias);
    return affine.linear(input, bias, contiguous_view(workspace, affine.shape()));
}
Tensor ProjectionWeight::packed_linear(const Tensor& input, const std::vector<int>& widths,
                                       const Tensor& bias, const Tensor& workspace) const {
    if (!affine.defined()) return elman::packed_linear(input, dense, widths, bias);
    return affine.packed_linear(input, widths, bias, contiguous_view(workspace, affine.shape()));
}
Tensor ProjectionWeight::embedding(const Tensor& indices, DType dtype) const {
    return affine.defined() ? affine.embedding(indices, dtype) : index_select(dense, 0, indices);
}
ProjectionWeight ProjectionWeight::concat_rows(const std::vector<ProjectionWeight>& weights) {
    if (weights.empty()) throw std::invalid_argument("Projection weight: empty projection packing");
    const bool quantized = weights[0].affine.defined();
    std::vector<Tensor> dense;
    std::vector<QuantizedTensor> affine;
    for (const auto& weight : weights) {
        if (weight.affine.defined() != quantized)
            throw std::invalid_argument("Projection weight: packed partitions must share dense/affine storage");
        if (quantized) affine.push_back(weight.affine);
        else dense.push_back(weight.dense);
    }
    return quantized ? ProjectionWeight(QuantizedTensor::concat_rows(affine)) : ProjectionWeight(concat(dense, 0));
}
} // namespace elman
