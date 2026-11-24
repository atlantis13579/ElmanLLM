#include "IO/CTranslate2.h"

#include <cmath>
#include <set>

#include "Core/TensorDispatch.h"
#include "IO/BinaryInput.h"

namespace elman {
namespace {
using io::BinaryInput;
using io::require;

std::string read_name(BinaryInput& input) {
    const auto count = input.le(2);
    require(count > 1, "Empty CTranslate2 name");
    std::string value = input.text(count);
    require(value.back() == '\0' && value.find('\0') == value.size() - 1,
            "Invalid CTranslate2 string terminator");
    value.pop_back();
    return value;
}

DType storage_type(uint64_t id) {
    switch (id) {
        case 0: return DType::Float32;
        case 1: return DType::Int8;
        case 2: return DType::Int16;
        case 3: return DType::Int32;
        case 4: return DType::Float16;
        case 5: return DType::BFloat16;
        default: throw std::invalid_argument("Unsupported CTranslate2 storage type");
    }
}
}

CTranslate2File::CTranslate2File(const std::filesystem::path& path) : path_(path) {
    BinaryInput input(path);
    require(input.le(4) == 6, "Unsupported CTranslate2 binary version (expected 6)");
    specification_ = read_name(input);
    revision_ = static_cast<uint32_t>(input.le(4));
    const uint64_t count = input.le(4);
    require(count > 0 && count <= input.remaining() / 10, "Invalid CTranslate2 tensor count");
    for (uint64_t i = 0; i < count; ++i) {
        const std::string name = read_name(input);
        const uint64_t rank = input.le(1);
        require(rank <= 8, "CTranslate2 tensor rank exceeds eight dimensions");
        Shape shape(static_cast<size_t>(rank));
        for (int& dimension : shape) {
            const uint64_t value = input.le(4);
            require(value > 0 && value <= uint64_t(std::numeric_limits<int>::max()),
                    "Invalid CTranslate2 tensor dimension");
            dimension = static_cast<int>(value);
        }
        const DType dtype = storage_type(input.le(1));
        const uint64_t bytes = input.le(4);
        require(bytes == dtype_storage_bytes(dtype, shape_element_count(shape)),
                "CTranslate2 tensor storage size does not match its shape");
        require(tensors_.emplace(name, SafetensorInfo{dtype, shape, input.position(), bytes}).second,
                "Duplicate CTranslate2 tensor name");
        input.skip(bytes);
    }
    const uint64_t alias_count = input.le(4);
    require(alias_count <= input.remaining() / 8, "Invalid CTranslate2 alias count");
    std::map<std::string, std::string> aliases;
    for (uint64_t i = 0; i < alias_count; ++i) {
        const std::string alias = read_name(input);
        const std::string target = read_name(input);
        require(!tensors_.count(alias) && aliases.emplace(alias, target).second,
                "Duplicate CTranslate2 alias name");
    }
    require(input.remaining() == 0, "Unexpected trailing CTranslate2 data");
    for (const auto& entry : aliases) {
        std::string target = entry.second;
        std::set<std::string> visited{entry.first};
        while (!tensors_.count(target)) {
            require(visited.insert(target).second, "Cyclic CTranslate2 aliases");
            const auto found = aliases.find(target);
            require(found != aliases.end(), "CTranslate2 alias target is missing");
            target = found->second;
        }
        tensors_.emplace(entry.first, tensors_.at(target));
    }
}

const SafetensorInfo& CTranslate2File::info(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) throw std::invalid_argument("Missing CTranslate2 tensor: " + name);
    return found->second;
}

Tensor CTranslate2File::load(const std::string& name, Backend backend) const {
    const auto& entry = info(name);
    BinaryInput input(path_);
    input.seek(entry.offset);
    return Tensor::from_raw(entry.shape, entry.dtype, input.bytes(entry.bytes), backend);
}

Tensor CTranslate2File::load_float(const std::string& name, Backend backend, DType dtype) const {
    require(is_compute_float(dtype), "CTranslate2 weights require a compute floating dtype");
    const auto& entry = info(name);
    if (is_compute_float(entry.dtype)) return cast(load(name, backend), dtype);
    require(entry.dtype == DType::Int8 || entry.dtype == DType::Int16,
            "Unsupported CTranslate2 quantized weight type");
    const auto& scale_info = info(name + "_scale");
    require(is_compute_float(scale_info.dtype), "CTranslate2 quantization scale must be floating point");
    const auto scales = load(name + "_scale").data();
    require(!entry.shape.empty() && (scales.size() == 1 || scales.size() == size_t(entry.shape.front())),
            "CTranslate2 quantization scale must be scalar or per-row");
    for (float scale : scales) require(std::isfinite(scale) && scale > 0,
                                      "Invalid CTranslate2 quantization scale");
    Shape scale_shape(entry.shape.size(), 1);
    scale_shape.front() = static_cast<int>(scales.size());
    Tensor weight = cast(load(name, backend), DType::Float32);
    Tensor scale = Tensor::from_data(scale_shape, scales, backend);
    return cast(div(weight, scale), dtype);
}

} // namespace elman
