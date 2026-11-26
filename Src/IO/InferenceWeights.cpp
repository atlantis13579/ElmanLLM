#include "IO/InferenceWeights.h"
#include "IO/Json.h"
#include "Core/Memory.h"
#include "Core/TensorDispatch.h"
#include "Kernals/affine_quantization.h"
#include <algorithm>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const std::string& message) {
    if (!valid) throw std::invalid_argument("Inference weights: " + message);
}
DType code_dtype(const std::string& name) {
    if (name == "qint8") return DType::QInt8;
    if (name == "quint8") return DType::QUInt8;
    if (name == "qint32") return DType::QInt32;
    if (name == "quint4") return DType::QUInt4;
    if (name == "quint2") return DType::QUInt2;
    throw std::invalid_argument("Inference weights: unsupported affine code dtype");
}
}
InferenceWeights::InferenceWeights(const std::string& directory) : weights_(directory) {
    const std::filesystem::path metadata = std::filesystem::path(directory) / "quantization.json";
    if (!std::filesystem::exists(metadata)) { tensors_ = weights_.tensors(); return; }
    const Json document = Json::read(metadata.string());
    require(document.at("format").string() == "elman-affine-v1", "unsupported quantization format");
    std::set<std::string> physical;
    for (const std::pair<const std::string, Json>& item : document.at("tensors").object()) {
        const Json& spec = item.second;
        SafetensorInfo info;
        info.dtype = code_dtype(spec.at("dtype").string());
        for (const Json& dimension : spec.at("shape").array()) info.shape.push_back(dimension.integer());
        require(info.shape.size() <= 8, "logical weight rank exceeds eight");
        const size_t count = shape_element_count(info.shape);
        info.bytes = dtype_storage_bytes(info.dtype, count);
        AffineInfo affine{spec.at("codes").string(), spec.at("scales").string(), spec.at("zero_points").string(), spec.at("group_size").integer()};
        require(affine.group_size > 0, "group size must be positive");
        for (const std::string* name : {&affine.codes, &affine.scales, &affine.zeros})
            require(physical.insert(*name).second, "physical payload is reused by multiple affine roles");
        const SafetensorInfo& codes = weights_.info(affine.codes);
        require(codes.dtype == DType::UInt8 && codes.shape.size() == 1 && codes.bytes == info.bytes,
                "affine code byte count or storage dtype differs");
        const SafetensorInfo& scales = weights_.info(affine.scales);
        const SafetensorInfo& zeros = weights_.info(affine.zeros);
        require(is_arithmetic_float(scales.dtype), "scales require real floating storage");
        require(zeros.dtype == DType::Int32 || zeros.dtype == DType::Int64, "zero points require integer storage");
        const int columns = info.shape.empty() ? 1 : info.shape.back();
        const size_t rows = info.shape.empty() ? size_t(1) : shape_element_count(Shape(info.shape.begin(), info.shape.end() - 1));
        const size_t target[] = {rows, size_t((int64_t(columns) + affine.group_size - 1) / affine.group_size)};
        for (const SafetensorInfo* parameter : {&scales, &zeros}) {
            require(parameter->shape.size() <= 2, "affine metadata rank exceeds two");
            for (size_t d = 0; d < parameter->shape.size(); ++d)
                require(parameter->shape[d] == 1 || size_t(parameter->shape[d]) == target[d + 2 - parameter->shape.size()],
                        "affine metadata does not broadcast to row groups");
        }
        tensors_.emplace(item.first, info);
        quantized_.emplace(item.first, std::move(affine));
    }
    require(!quantized_.empty(), "empty affine catalog");
    for (const std::pair<const std::string, SafetensorInfo>& item : weights_.tensors()) {
        if (physical.count(item.first)) continue;
        require(tensors_.emplace(item.first, item.second).second, "dense and affine logical names overlap");
    }
}
const SafetensorInfo& InferenceWeights::info(const std::string& name) const {
    const auto found = tensors_.find(name);
    require(found != tensors_.end(), "missing logical tensor " + name);
    return found->second;
}
Tensor InferenceWeights::load(const std::string& name, Backend backend) const {
    info(name);
    require(!quantized(name), "affine tensor requires its scale metadata");
    return weights_.load(name, backend);
}
Tensor InferenceWeights::load_slice(const std::string& name, int axis, int offset, int count, Backend backend) const {
    info(name);
    require(!quantized(name), "tensor-parallel slicing requires dense weights");
    return weights_.load_slice(name, axis, offset, count, backend);
}
QuantizedTensor InferenceWeights::load_quantized(const std::string& name, Backend backend) const {
    const SafetensorInfo& logical = info(name);
    require(quantized(name), "requested tensor has no affine metadata");
    const InferenceWeights::AffineInfo& affine = quantized_.at(name);
    Tensor codes = weights_.load_storage(affine.codes, logical.shape, logical.dtype, backend);
    Tensor scales = weights_.load(affine.scales, backend);
    Tensor zeros = weights_.load(affine.zeros, backend);
    return QuantizedTensor(std::move(codes), std::move(scales), std::move(zeros), affine.group_size);
}
size_t InferenceWeights::resident_bytes(Backend backend, DType dtype) const {
    size_t result = 0;
    for (const std::pair<const std::string, SafetensorInfo>& item : tensors_) result = checked_memory_add(result, resident_bytes(item.first, backend, dtype));
    return result;
}
size_t InferenceWeights::resident_bytes(const std::string& name, Backend backend, DType dtype) const {
    const SafetensorInfo& logical = info(name);
    const auto found = quantized_.find(name);
    if (found == quantized_.end()) {
        require(is_compute_float(logical.dtype), "dense inference tensor requires floating storage");
        return memory_allocation_bytes(backend, dtype_storage_bytes(dtype, shape_element_count(logical.shape)));
    }
    size_t result = 0;
    for (const std::string* payload : {&found->second.codes, &found->second.scales, &found->second.zeros})
        result = checked_memory_add(result, memory_allocation_bytes(backend, size_t(weights_.info(*payload).bytes)));
    return result;
}
size_t InferenceWeights::largest_payload() const {
    size_t result = 0;
    for (const std::pair<const std::string, SafetensorInfo>& item : weights_.tensors()) result = std::max(result, size_t(item.second.bytes));
    return result;
}
size_t InferenceWeights::largest_quantized_weight() const {
    size_t result = 0;
    for (const std::pair<const std::string, InferenceWeights::AffineInfo>& item : quantized_) result = std::max(result, shape_element_count(info(item.first).shape));
    return result;
}
} // namespace elman
