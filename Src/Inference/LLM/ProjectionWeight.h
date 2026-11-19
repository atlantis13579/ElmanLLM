#pragma once

#include "Core/Quantization.h"

namespace elman {
// Dense or affine inference weight with caller-owned dequantization workspace.
struct ProjectionWeight {
    Tensor dense;
    QuantizedTensor affine;
    ProjectionWeight() = default;
    explicit ProjectionWeight(Tensor value) : dense(std::move(value)) {}
    explicit ProjectionWeight(QuantizedTensor value) : affine(std::move(value)) {}
    size_t workspace_count() const { return affine.defined() ? affine.numel() : 0; }
    Tensor materialize(const Tensor& workspace) const;
    ProjectionWeight to_device(int ordinal) const;
    Tensor linear(const Tensor& input, const Tensor& bias, const Tensor& workspace) const;
    Tensor packed_linear(const Tensor& input, const std::vector<int>& widths,
                         const Tensor& bias, const Tensor& workspace) const;
    Tensor embedding(const Tensor& indices, DType dtype) const;
    static ProjectionWeight concat_rows(const std::vector<ProjectionWeight>& weights);
};
} // namespace elman
