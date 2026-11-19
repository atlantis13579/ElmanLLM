#pragma once

#include "Core/Tensor.h"
#include <cstdint>
#include <limits>
#include <mutex>

namespace elman {

// Affine codes carry their metadata explicitly. Groups partition the final
// dimension independently in every flattened row, including partial groups.
class QuantizedTensor {
public:
    QuantizedTensor() = default;
    QuantizedTensor(Tensor codes, Tensor scales, Tensor zero_points, int group_size);
    static QuantizedTensor quantize(const Tensor& input, DType storage_dtype,
                                     const Tensor& scales, const Tensor& zero_points,
                                     int group_size);
    static QuantizedTensor concat_rows(const std::vector<QuantizedTensor>& values);

    bool defined() const { return codes_.defined(); }
    const Shape& shape() const { return codes_.shape(); }
    Backend backend() const { return codes_.backend(); }
    DType storage_dtype() const { return codes_.dtype(); }
    int group_size() const { return group_size_; }
    size_t numel() const { return codes_.numel(); }
    const Tensor& codes() const { return codes_; }
    const Tensor& scales() const { return scales_; }
    const Tensor& zero_points() const { return zero_points_; }

    Tensor dequantize(DType dtype = DType::Float32) const;
    void dequantize_into(const Tensor& output) const;
    Tensor embedding(const Tensor& indices, DType dtype) const;
    Tensor linear(const Tensor& input, const Tensor& bias = {}, const Tensor& workspace = {}) const;
    Tensor packed_linear(const Tensor& input, const std::vector<int>& widths,
                         const Tensor& bias = {}, const Tensor& workspace = {}) const;
    void validate() const;

private:
    Tensor codes_, scales_, zero_points_;
    int group_size_ = 0;
    struct Validation {
        std::mutex mutex;
        uint64_t scale_version = std::numeric_limits<uint64_t>::max();
        uint64_t zero_version = std::numeric_limits<uint64_t>::max();
    };
    std::shared_ptr<Validation> validation_ = std::make_shared<Validation>();
};

QuantizedTensor quantized_add(const QuantizedTensor& a, const QuantizedTensor& b,
                             DType output_dtype, const Tensor& scales,
                             const Tensor& zero_points, int group_size);
QuantizedTensor quantized_mul(const QuantizedTensor& a, const QuantizedTensor& b,
                             DType output_dtype, const Tensor& scales,
                             const Tensor& zero_points, int group_size);
QuantizedTensor quantized_relu(const QuantizedTensor& input);

} // namespace elman
