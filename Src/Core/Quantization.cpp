#include "Core/Quantization.h"
#include "Core/Graph.h"
#include "Core/TensorDispatch.h"
#include "Operators/affine_quantization.h"
#include <cmath>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}
Tensor allocate(const Shape& shape, DType dtype, Backend backend) {
    auto node = std::make_shared<TensorNode>();
    node->shape = shape;
    node->value = backend_allocate(backend, shape_element_count(shape), dtype);
    return Tensor(node);
}
Shape metadata_shape(const Shape& shape, int group_size) {
    require(group_size > 0, "Quantization group size must be positive");
    const int columns = shape.empty() ? 1 : shape.back();
    const auto rows = shape.empty() ? 1 : shape_element_count(Shape(shape.begin(), shape.end() - 1));
    const int groups = int((int64_t(columns) + group_size - 1) / group_size);
    require(rows <= size_t(std::numeric_limits<int>::max()), "Quantization row count exceeds Int32 indexing");
    return {int(rows), groups};
}
void metadata_dimensions(const Tensor& value, const Shape& target) {
    require(value.shape().size() <= 2, "Quantization metadata rank exceeds two");
    const int delta = 2 - int(value.shape().size());
    for (int d = 0; d < int(value.shape().size()); ++d)
        require(value.shape()[d] == 1 || value.shape()[d] == target[d + delta],
                "Quantization metadata does not broadcast to row groups");
}
std::vector<int64_t> zero_values(const Tensor& tensor) {
    if (tensor.dtype() == DType::Int64) return tensor.int64_data();
    const auto values = tensor.int_data();
    return std::vector<int64_t>(values.begin(), values.end());
}
}

QuantizedTensor::QuantizedTensor(Tensor codes, Tensor scales, Tensor zero_points, int group_size)
    : codes_(std::move(codes)), scales_(std::move(scales)), zero_points_(std::move(zero_points)), group_size_(group_size) {
    validate();
}
void QuantizedTensor::validate() const {
    require(codes_.defined() && scales_.defined() && zero_points_.defined(), "Undefined quantization storage or metadata");
    require(quantized_storage_bits(codes_.dtype()) != 0, "Affine quantization requires a quantized storage dtype");
    require(is_arithmetic_float(scales_.dtype()), "Quantization scales require real floating storage");
    require(zero_points_.dtype() == DType::Int32 || zero_points_.dtype() == DType::Int64,
            "Quantization zero points require Int32 or Int64 storage");
    require(codes_.backend() == scales_.backend() && codes_.backend() == zero_points_.backend(),
            "Quantization metadata backend differs from its codes");
    require(!codes_.requires_grad() && !scales_.requires_grad() && !zero_points_.requires_grad(),
            "Quantized values and metadata do not carry differentiable state");
    const auto target = metadata_shape(codes_.shape(), group_size_);
    metadata_dimensions(scales_, target);
    metadata_dimensions(zero_points_, target);
    // Public mutations increment buffer versions. Revalidate changed metadata,
    // while immutable inference weights avoid repeated device readback.
    std::lock_guard<std::mutex> lock(validation_->mutex);
    if (validation_->scale_version != scales_.node()->value->version) {
        if (scales_.dtype() == DType::Float64) {
            for (double value : scales_.double_data()) require(std::isfinite(value) && value > 0, "Quantization scale must be finite and positive");
        } else {
            for (float value : scales_.data()) require(std::isfinite(value) && value > 0, "Quantization scale must be finite and positive");
        }
        validation_->scale_version = scales_.node()->value->version;
    }
    if (validation_->zero_version != zero_points_.node()->value->version) {
        for (int64_t value : zero_values(zero_points_))
            require(value >= quantized_minimum(codes_.dtype()) && value <= quantized_maximum(codes_.dtype()),
                    "Quantization zero point is outside the code range");
        validation_->zero_version = zero_points_.node()->value->version;
    }
}
QuantizedTensor QuantizedTensor::quantize(const Tensor& input, DType dtype, const Tensor& scales,
                                          const Tensor& zero_points, int group_size) {
    require(is_arithmetic_float(input.dtype()), "Quantization requires real floating input");
    auto codes = allocate(input.shape(), dtype, input.backend());
    QuantizedTensor result(codes, scales, zero_points, group_size);
    dispatch_tensor_into("affine_quantize", codes, {input, scales, zero_points}, {group_size});
    return result;
}
Tensor QuantizedTensor::dequantize(DType dtype) const {
    require(is_arithmetic_float(dtype), "Dequantization requires a real floating output dtype");
    validate();
    return dispatch_tensor("affine_dequantize", shape(), {codes_, scales_, zero_points_}, {group_size_}, dtype);
}
void QuantizedTensor::dequantize_into(const Tensor& output) const {
    require(output.shape() == shape() && output.backend() == backend() && is_arithmetic_float(output.dtype()),
            "Dequantization workspace shape, dtype or backend mismatch");
    require(!output.requires_grad(), "Dequantization cannot mutate differentiable workspace");
    validate();
    dispatch_tensor_into("affine_dequantize", output, {codes_, scales_, zero_points_}, {group_size_});
    ++output.node()->value->version;
}
Tensor QuantizedTensor::embedding(const Tensor& indices, DType dtype) const {
    require(shape().size() == 2 && indices.backend() == backend() && is_arithmetic_float(dtype),
            "Invalid quantized embedding shape, dtype or backend");
    require(indices.dtype() == DType::Int32 || indices.dtype() == DType::Int64, "Quantized embedding indices require integers");
    validate();
    for (int64_t id : zero_values(indices)) require(id >= 0 && id < shape()[0], "Quantized embedding index out of range");
    auto result_shape = indices.shape();
    result_shape.push_back(shape()[1]);
    return dispatch_tensor("affine_embedding", result_shape, {codes_, scales_, zero_points_, indices}, {group_size_}, dtype);
}
Tensor QuantizedTensor::linear(const Tensor& input, const Tensor& bias, const Tensor& workspace) const {
    Tensor weight;
    if (workspace.defined()) {
        require(!input.requires_grad(), "Reusable quantized workspace requires inference input");
        require(workspace.dtype() == input.dtype(), "Quantized linear workspace dtype mismatch");
        dequantize_into(workspace);
        weight = workspace;
    } else weight = dequantize(input.dtype());
    return elman::linear(input, weight, bias);
}
Tensor QuantizedTensor::packed_linear(const Tensor& input, const std::vector<int>& widths,
                                      const Tensor& bias, const Tensor& workspace) const {
    Tensor weight;
    if (workspace.defined()) {
        require(!input.requires_grad(), "Reusable quantized workspace requires inference input");
        require(workspace.dtype() == input.dtype(), "Quantized packed-linear workspace dtype mismatch");
        dequantize_into(workspace);
        weight = workspace;
    } else weight = dequantize(input.dtype());
    return elman::packed_linear(input, weight, widths, bias);
}
QuantizedTensor QuantizedTensor::concat_rows(const std::vector<QuantizedTensor>& values) {
    require(!values.empty() && values[0].shape().size() == 2, "Quantized row concatenation requires matrices");
    const auto& first = values[0];
    std::vector<Tensor> codes, scales, zeros;
    bool constant_zero = first.zero_points_.numel() == 1;
    int64_t zero = constant_zero ? zero_values(first.zero_points_)[0] : 0;
    for (const auto& value : values) {
        value.validate();
        require(value.shape().size() == 2 && value.shape()[1] == first.shape()[1] && value.backend() == first.backend() &&
                value.storage_dtype() == first.storage_dtype() && value.group_size_ == first.group_size_ &&
                value.scales_.dtype() == first.scales_.dtype() && value.zero_points_.dtype() == first.zero_points_.dtype(),
                "Quantized row concatenation metadata mismatch");
        codes.push_back(value.codes_);
        const auto target = metadata_shape(value.shape(), value.group_size_);
        scales.push_back(expand(value.scales_, target));
        zeros.push_back(value.zero_points_);
        constant_zero = constant_zero && value.zero_points_.numel() == 1 && zero_values(value.zero_points_)[0] == zero;
    }
    Tensor joined_zero = first.zero_points_;
    if (!constant_zero) {
        for (size_t i = 0; i < zeros.size(); ++i) zeros[i] = expand(zeros[i], metadata_shape(values[i].shape(), first.group_size_));
        joined_zero = concat(zeros, 0);
    }
    return QuantizedTensor(concat(codes, 0), concat(scales, 0), joined_zero, first.group_size_);
}
QuantizedTensor quantized_add(const QuantizedTensor& a, const QuantizedTensor& b, DType dtype,
                             const Tensor& scales, const Tensor& zeros, int group_size) {
    return QuantizedTensor::quantize(add(a.dequantize(), b.dequantize()), dtype, scales, zeros, group_size);
}
QuantizedTensor quantized_mul(const QuantizedTensor& a, const QuantizedTensor& b, DType dtype,
                             const Tensor& scales, const Tensor& zeros, int group_size) {
    return QuantizedTensor::quantize(mul(a.dequantize(), b.dequantize()), dtype, scales, zeros, group_size);
}
QuantizedTensor quantized_relu(const QuantizedTensor& input) {
    input.validate();
    auto codes = dispatch_tensor("affine_relu", input.shape(), {input.codes(), input.scales(), input.zero_points()},
                                  {input.group_size()}, input.storage_dtype());
    return QuantizedTensor(codes, input.scales(), input.zero_points(), input.group_size());
}
} // namespace elman
