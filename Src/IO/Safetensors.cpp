#include "IO/Safetensors.h"
#include "Core/Memory.h"
#include "IO/AtomicFile.h"
#include "IO/Json.h"
#include "Core/TensorDispatch.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("Safetensors: " + message);
}
uint64_t unsigned_integer(const Json& value) {
    double number = value.number();
    require(std::isfinite(number) && number >= 0 && number <= 9007199254740991.0 &&
            std::floor(number) == number, "expected an exact nonnegative integer");
    return static_cast<uint64_t>(number);
}
const std::map<std::string, DType>& storage_dtypes() {
    static const std::map<std::string, DType> types = {
        {"BOOL", DType::Bool}, {"I8", DType::Int8}, {"U8", DType::UInt8},
        {"I16", DType::Int16}, {"U16", DType::UInt16},
        {"I32", DType::Int32}, {"U32", DType::UInt32},
        {"I64", DType::Int64}, {"U64", DType::UInt64},
        {"F16", DType::Float16}, {"BF16", DType::BFloat16},
        {"F32", DType::Float32}, {"F64", DType::Float64},
        {"C64", DType::Complex64},
        {"F8_E4M3", DType::Float8E4M3FN}, {"F8_E5M2", DType::Float8E5M2}
    };
    return types;
}
DType storage_dtype(const std::string& name) {
    const std::map<std::string, DType>& types = storage_dtypes();
    auto found = types.find(name);
    require(found != types.end(), "unsupported dtype " + name);
    return found->second;
}
void validate_utf8(const std::string& text) {
    for (size_t i = 0; i < text.size();) {
        unsigned char first = static_cast<unsigned char>(text[i++]);
        if (first < 0x80) continue;
        int count = first >= 0xc2 && first <= 0xdf ? 1 :
                    first >= 0xe0 && first <= 0xef ? 2 :
                    first >= 0xf0 && first <= 0xf4 ? 3 : -1;
        require(count >= 0 && i + count <= text.size(), "invalid UTF-8 header");
        uint32_t code = first & ((1u << (6 - count)) - 1);
        for (int j = 0; j < count; ++j) {
            unsigned char byte = static_cast<unsigned char>(text[i++]);
            require((byte & 0xc0) == 0x80, "invalid UTF-8 continuation");
            code = (code << 6) | (byte & 0x3f);
        }
        uint32_t minimum = count == 1 ? 0x80 : count == 2 ? 0x800 : 0x10000;
        require(code >= minimum && code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff),
                "invalid UTF-8 code point");
    }
}
void validate_integer_tokens(const std::string& text) {
    bool quoted = false;
    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        if (quoted) {
            if (ch == '\\') ++i;
            else if (ch == '"') quoted = false;
        } else if (ch == '"') quoted = true;
        else if (ch == '-' || (ch >= '0' && ch <= '9')) {
            require(ch != '-', "negative header integer");
            while (i + 1 < text.size() && text[i + 1] >= '0' && text[i + 1] <= '9') ++i;
            require(i + 1 == text.size() || (text[i + 1] != '.' && text[i + 1] != 'e' && text[i + 1] != 'E'),
                    "header dimensions and offsets must use integer tokens");
        }
    }
}
uint64_t file_size(std::ifstream& file) {
    file.seekg(0, std::ios::end);
    std::streampos end = file.tellg();
    require(end >= 0, "cannot determine file size");
    file.seekg(0);
    return static_cast<uint64_t>(end);
}
void read_exact(std::ifstream& file, void* data, uint64_t bytes) {
    require(bytes <= uint64_t(std::numeric_limits<std::streamsize>::max()), "read exceeds stream limit");
    if (!bytes) return;
    file.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    require(bool(file), "truncated or unreadable file");
}
}

void write_safetensors_header(std::ostream& stream, const std::vector<SafetensorSpec>& entries,
                              const std::map<std::string, std::string>& metadata) {
    Json::Object catalog;
    if (!metadata.empty()) {
        Json::Object values;
        for (const std::pair<const std::string, std::string>& entry : metadata) {
            const std::string& key = entry.first;
            const std::string& value = entry.second;
            values.emplace(key, value);
        }
        catalog.emplace("__metadata__", std::move(values));
    }
    uint64_t offset = 0;
    for (const SafetensorSpec& spec : entries) {
        const std::string& name = spec.name;
        const DType dtype = spec.dtype;
        const Shape& dimensions = spec.shape;
        require(!name.empty() && name != "__metadata__" && !catalog.count(name), "invalid or duplicate output tensor name");
        const std::map<std::string, DType>& types = storage_dtypes();
        const auto type = std::find_if(types.begin(), types.end(), [&](const std::pair<const std::string, DType>& item) { return item.second == dtype; });
        require(type != types.end(), std::string("cannot save dtype ") + dtype_name(dtype));
        const uint64_t bytes = dtype_storage_bytes(dtype, shape_element_count(dimensions));
        require(bytes <= 9007199254740991ull - offset, "output offsets exceed exact JSON integer range");
        Json::Array shape;
        for (int dim : dimensions) shape.emplace_back(dim);
        catalog.emplace(name, Json::Object{{"dtype", type->first}, {"shape", std::move(shape)},
                                          {"data_offsets", Json::Array{double(offset), double(offset + bytes)}}});
        offset += bytes;
    }
    std::string header = Json(std::move(catalog)).dump(0);
    validate_utf8(header);
    header.append((8 - header.size() % 8) % 8, ' ');
    require(header.size() <= 100000000, "output header exceeds format limit");
    for (int byte = 0; byte < 8; ++byte) stream.put(char(uint64_t(header.size()) >> (byte * 8)));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    require(bool(stream), "cannot write header");
}

void save_safetensors(const std::filesystem::path& path, const std::map<std::string, Tensor>& tensors) {
    std::vector<SafetensorSpec> entries;
    entries.reserve(tensors.size());
    for (const std::pair<const std::string, Tensor>& entry : tensors) {
        const std::string& name = entry.first;
        const Tensor& tensor = entry.second;
        require(tensor.defined(), "invalid output tensor value");
        entries.push_back({name, tensor.dtype(), tensor.shape()});
    }
    write_file_atomically(path, [&](const std::filesystem::path& temporary) {
        std::ofstream file(temporary, std::ios::binary);
        require(bool(file), "cannot create output " + path.u8string());
        write_safetensors_header(file, entries);
        for (const std::pair<const std::string, Tensor>& entry : tensors) {
            std::vector<uint8_t> bytes = entry.second.raw_data();
            if (!bytes.empty()) file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            require(bool(file), "cannot write tensor " + entry.first);
        }
        file.close();
        require(bool(file), "cannot close output " + path.u8string());
    });
}

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : path_(path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "cannot open " + path.u8string());
    file_size_ = file_size(file);
    require(file_size_ >= 8, "missing header length");
    uint8_t prefix[8];
    read_exact(file, prefix, 8);
    uint64_t header_bytes = 0;
    for (int i = 0; i < 8; ++i) header_bytes |= uint64_t(prefix[i]) << (i * 8);
    require(header_bytes > 0 && header_bytes <= 100000000, "invalid or oversized header");
    require(header_bytes <= file_size_ - 8, "header exceeds file size");
    std::string header(size_t(header_bytes), '\0');
    read_exact(file, header.data(), header_bytes);
    require(header.front() == '{', "header must start with an object");
    validate_utf8(header);
    validate_integer_tokens(header);
    Json root = Json::parse(header);
    data_start_ = 8 + header_bytes;
    uint64_t payload_bytes = file_size_ - data_start_;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const std::pair<const std::string, Json>& entry : root.object()) {
        if (entry.first == "__metadata__") {
            for (const std::pair<const std::string, Json>& item : entry.second.object()) metadata_.emplace(item.first, item.second.string());
            continue;
        }
        const Json& value = entry.second;
        SafetensorInfo tensor;
        tensor.dtype = storage_dtype(value.at("dtype").string());
        const Json::Array& dimensions = value.at("shape").array();
        require(dimensions.size() <= 8, "tensor rank exceeds the runtime limit of eight");
        for (const Json& dim : dimensions) {
            uint64_t size = unsigned_integer(dim);
            require(size <= uint64_t(std::numeric_limits<int>::max()), "dimension exceeds runtime indexing limit");
            tensor.shape.push_back(static_cast<int>(size));
        }
        size_t elements = shape_element_count(tensor.shape);
        const Json::Array& offsets = value.at("data_offsets").array();
        require(offsets.size() == 2, "expected two data offsets");
        uint64_t begin = unsigned_integer(offsets[0]), end = unsigned_integer(offsets[1]);
        require(begin <= end && end <= payload_bytes, "invalid tensor offsets");
        require(end - begin == dtype_storage_bytes(tensor.dtype, elements), "shape and payload byte count differ");
        tensor.offset = begin;
        tensor.bytes = end - begin;
        tensors_.emplace(entry.first, std::move(tensor));
        ranges.emplace_back(begin, end);
    }
    std::sort(ranges.begin(), ranges.end());
    uint64_t end = 0;
    for (const std::pair<uint64_t, uint64_t>& range : ranges) {
        require(range.first == end, "payload contains a hole or overlap");
        end = range.second;
    }
    require(end == payload_bytes, "unindexed bytes at the end of the file");
}

const SafetensorInfo& SafetensorsFile::info(const std::string& name) const {
    auto found = tensors_.find(name);
    require(found != tensors_.end(), "missing tensor " + name);
    return found->second;
}
void SafetensorsFile::read_bytes(const std::string& name, uint64_t offset, void* output, size_t count) const {
    const SafetensorInfo& tensor = info(name);
    require(offset <= tensor.bytes && count <= tensor.bytes - offset && (!count || output), "invalid payload range for " + name);
    if (!count) return;
    std::ifstream file(path_, std::ios::binary);
    require(bool(file) && file_size(file) == file_size_, "cannot reopen tensor file " + path_.u8string());
    file.seekg(static_cast<std::streamoff>(data_start_ + tensor.offset + offset));
    read_exact(file, output, count);
}
Tensor SafetensorsFile::load(const std::string& name, Backend backend) const {
    const SafetensorInfo& tensor = info(name);
    return load_storage(name, tensor.shape, tensor.dtype, backend);
}
Tensor SafetensorsFile::load_slice(const std::string& name, int axis, int offset, int count, Backend backend) const {
    const SafetensorInfo& tensor = info(name);
    if (axis < 0 || axis > 1 || size_t(axis) >= tensor.shape.size() || tensor.shape.size() > 2 ||
        offset < 0 || count < 0 || offset > tensor.shape[axis] || count > tensor.shape[axis] - offset ||
        dtype_info(tensor.dtype).bits % 8 != 0) throw std::invalid_argument("Unsupported Safetensors slice for " + name);
    if (count == tensor.shape[axis]) return load(name, backend);
    const uint64_t element_bytes = dtype_info(tensor.dtype).bits / 8;
    const uint64_t columns = tensor.shape.size() == 2 ? tensor.shape[1] : 1;
    Shape shape = tensor.shape;
    shape[axis] = count;
    const size_t bytes_count = dtype_storage_bytes(tensor.dtype, shape_element_count(shape));
    MemoryReservation reservation(Backend::CPU, bytes_count);
    std::vector<uint8_t> bytes(bytes_count);
    std::ifstream file(path_, std::ios::binary);
    require(bool(file) && file_size(file) == file_size_, "cannot reopen tensor slice file");
    if (!bytes.empty()) {
        const uint64_t base = data_start_ + tensor.offset;
        if (axis == 0) {
            file.seekg(static_cast<std::streamoff>(base + uint64_t(offset) * columns * element_bytes));
            read_exact(file, bytes.data(), bytes.size());
        } else {
            const size_t row_bytes = size_t(count) * element_bytes;
            for (int row = 0; row < shape[0]; ++row) {
                file.seekg(static_cast<std::streamoff>(base + (uint64_t(row) * columns + offset) * element_bytes));
                read_exact(file, bytes.data() + size_t(row) * row_bytes, row_bytes);
            }
        }
    }
    return Tensor::from_raw(std::move(shape), tensor.dtype, bytes, backend);
}
Tensor SafetensorsFile::load_storage(const std::string& name, const Shape& shape, DType dtype, Backend backend) const {
    const SafetensorInfo& tensor = info(name);
    require(dtype_storage_bytes(dtype, shape_element_count(shape)) == tensor.bytes, "logical and physical payload sizes differ");
    std::ifstream file(path_, std::ios::binary);
    require(bool(file), "cannot reopen " + path_.u8string());
    require(file_size(file) == file_size_, "checkpoint size changed after catalog validation");
    file.seekg(static_cast<std::streamoff>(data_start_ + tensor.offset));
    MemoryReservation reservation(Backend::CPU, static_cast<size_t>(tensor.bytes));
    std::vector<uint8_t> bytes(static_cast<size_t>(tensor.bytes));
    read_exact(file, bytes.data(), tensor.bytes);
    return Tensor::from_raw(shape, dtype, bytes, backend);
}

SafetensorsWeights::SafetensorsWeights(const std::filesystem::path& directory) {
    namespace fs = std::filesystem;
    fs::path root = fs::canonical(directory);
    fs::path index_path = root / "model.safetensors.index.json";
    if (!fs::exists(index_path)) {
        files_.emplace_back(root / "model.safetensors");
        tensors_ = files_.front().tensors();
        for (const std::pair<const std::string, SafetensorInfo>& item : tensors_) locations_.emplace(item.first, 0);
        return;
    }
    Json index = Json::read(index_path);
    std::map<std::string, size_t> shards;
    const Json::Object& weights = index.at("weight_map").object();
    require(!weights.empty(), "empty sharded weight map");
    for (const std::pair<const std::string, Json>& item : weights) {
        std::string filename = item.second.string();
        const std::filesystem::path relative = fs::u8path(filename);
        require(!filename.empty() && !relative.is_absolute() && !relative.has_root_path() &&
                relative.filename() == relative && filename != "." && filename != ".." &&
                filename.find(':') == std::string::npos && filename.find('\\') == std::string::npos &&
                filename.find('/') == std::string::npos, "invalid shard filename");
        fs::path shard = fs::canonical(root / relative);
        require(shard.parent_path() == root, "shard resolves outside the model directory");
        auto found = shards.find(filename);
        if (found == shards.end()) {
            size_t position = files_.size();
            files_.emplace_back(shard);
            found = shards.emplace(filename, position).first;
        }
        size_t position = found->second;
        tensors_.emplace(item.first, files_[position].info(item.first));
        locations_.emplace(item.first, position);
    }
    for (size_t i = 0; i < files_.size(); ++i) {
        for (const std::pair<const std::string, SafetensorInfo>& item : files_[i].tensors()) {
            auto found = locations_.find(item.first);
            require(found != locations_.end() && found->second == i,
                    "shard tensor is missing from the index or appears in multiple shards: " + item.first);
        }
    }
}
const SafetensorInfo& SafetensorsWeights::info(const std::string& name) const {
    auto found = tensors_.find(name);
    require(found != tensors_.end(), "missing weight " + name);
    return found->second;
}
void SafetensorsWeights::read_bytes(const std::string& name, uint64_t offset, void* output, size_t count) const {
    info(name);
    files_[locations_.at(name)].read_bytes(name, offset, output, count);
}
Tensor SafetensorsWeights::load(const std::string& name, Backend backend) const {
    info(name);
    return files_[locations_.at(name)].load(name, backend);
}
Tensor SafetensorsWeights::load_slice(const std::string& name, int axis, int offset, int count, Backend backend) const {
    info(name);
    return files_.at(locations_.at(name)).load_slice(name, axis, offset, count, backend);
}

Tensor SafetensorsWeights::load_storage(const std::string& name, const Shape& shape, DType dtype, Backend backend) const {
    info(name);
    return files_[locations_.at(name)].load_storage(name, shape, dtype, backend);
}
} // namespace elman
