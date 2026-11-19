#pragma once

#include "IO/Safetensors.h"
#include "Core/Quantization.h"

namespace elman {
// Optional affine metadata maps physical safetensors payloads to logical
// weights. Dense Hugging Face checkpoints retain their original catalog.
class InferenceWeights {
public:
    explicit InferenceWeights(const std::string& directory);
    const std::map<std::string, SafetensorInfo>& tensors() const { return tensors_; }
    const SafetensorInfo& info(const std::string& name) const;
    bool quantized(const std::string& name) const { return quantized_.count(name) != 0; }
    bool has_quantization() const { return !quantized_.empty(); }
    Tensor load(const std::string& name, Backend backend) const;
    Tensor load_slice(const std::string& name, int axis, int offset, int count, Backend backend) const;
    QuantizedTensor load_quantized(const std::string& name, Backend backend) const;
    size_t resident_bytes(Backend backend, DType dense_dtype) const;
    size_t resident_bytes(const std::string& name, Backend backend, DType dense_dtype) const;
    size_t largest_payload() const;
    size_t largest_quantized_weight() const;
private:
    struct AffineInfo { std::string codes, scales, zeros; int group_size = 0; };
    SafetensorsWeights weights_;
    std::map<std::string, SafetensorInfo> tensors_;
    std::map<std::string, AffineInfo> quantized_;
};
} // namespace elman
