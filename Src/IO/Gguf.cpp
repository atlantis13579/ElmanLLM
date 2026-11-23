#include "IO/Gguf.h"

#include <algorithm>
#include <cstring>
#include <set>

#include "Core/Float16.h"
#include "Core/TensorDispatch.h"
#include "IO/BinaryInput.h"

namespace elman {
namespace {

using io::BinaryInput;
using io::Bytes;
using io::require;

struct Encoding {
    DType dtype;
    uint32_t block, bytes;
};

Encoding encoding(uint32_t type) {
    switch (type) {
        case 0: return {DType::Float32, 1, 4};
        case 1: return {DType::Float16, 1, 2};
        case 2: return {DType::Float32, 32, 18};
        case 3: return {DType::Float32, 32, 20};
        case 6: return {DType::Float32, 32, 22};
        case 7: return {DType::Float32, 32, 24};
        case 8: return {DType::Float32, 32, 34};
        case 10: return {DType::Float32, 256, 84};
        case 11: return {DType::Float32, 256, 110};
        case 12: return {DType::Float32, 256, 144};
        case 13: return {DType::Float32, 256, 176};
        case 14: return {DType::Float32, 256, 210};
        case 24: return {DType::Int8, 1, 1};
        case 25: return {DType::Int16, 1, 2};
        case 26: return {DType::Int32, 1, 4};
        case 27: return {DType::Int64, 1, 8};
        case 28: return {DType::Float64, 1, 8};
        case 30: return {DType::BFloat16, 1, 2};
        default: throw std::invalid_argument("Unsupported GGUF tensor encoding " + std::to_string(type));
    }
}

size_t metadata_width(uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: throw std::invalid_argument("Unsupported GGUF metadata type " + std::to_string(type));
    }
}

template<typename T, typename Bits>
T bit_value(Bits bits) {
    static_assert(sizeof(T) == sizeof(Bits));
    T value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double number(BinaryInput& input, uint32_t type) {
    const uint64_t bits = input.le(metadata_width(type));
    switch (type) {
        case 1: return bit_value<int8_t>(uint8_t(bits));
        case 3: return bit_value<int16_t>(uint16_t(bits));
        case 5: return bit_value<int32_t>(uint32_t(bits));
        case 6: return bit_value<float>(uint32_t(bits));
        case 11: return double(bit_value<int64_t>(bits));
        case 12: return bit_value<double>(bits);
        default: return double(bits);
    }
}

void skip_array(BinaryInput& input) {
    const uint32_t type = uint32_t(input.le(4));
    const uint64_t count = input.le(8);
    if (type == 8) {
        require(count <= input.remaining() / 8, "Invalid GGUF metadata string array");
        for (uint64_t i = 0; i < count; ++i) input.skip(input.le(8));
    } else {
        const size_t width = metadata_width(type);
        require(count <= input.remaining() / width, "Invalid GGUF metadata array");
        input.skip(count * width);
    }
}

float half(Bytes bytes, size_t offset) { return numeric::half_to_float(uint16_t(bytes.le(offset, 2))); }

void decode_block(uint32_t type, Bytes bytes, float* values) {
    const uint8_t* q = bytes.data;
    if (type <= 8) {
        const float scale = half(bytes, 0);
        if (type == 8) {
            for (int i = 0; i < 32; ++i) values[i] = scale * bit_value<int8_t>(q[i + 2]);
            return;
        }
        const bool affine = type == 3 || type == 7;
        const bool five = type == 6 || type == 7;
        const float offset = affine ? half(bytes, 2) : 0.0f;
        const int header = affine ? 4 : 2;
        const uint32_t high = five ? uint32_t(bytes.le(header, 4)) : 0;
        for (int i = 0; i < 32; ++i) {
            const int code = ((q[header + (five ? 4 : 0) + i % 16] >> (4 * (i / 16))) & 15) |
                             int(((high >> i) & 1) << 4);
            values[i] = scale * (code - (affine ? 0 : five ? 16 : 8)) + offset;
        }
    } else if (type == 10) {
        const float scale = half(bytes, 80), minimum = half(bytes, 82);
        for (int i = 0; i < 256; ++i) {
            const int s = q[i / 16];
            const int code = (q[16 + 32 * (i / 128) + i % 32] >> (2 * (i % 128 / 32))) & 3;
            values[i] = scale * (s & 15) * code - minimum * (s >> 4);
        }
    } else if (type == 11) {
        const float scale = half(bytes, 108);
        for (int i = 0; i < 256; ++i) {
            const int group = i / 16;
            const int low = (q[96 + group % 8] >> (4 * (group / 8))) & 15;
            const int high = (q[104 + group % 4] >> (2 * (group / 4))) & 3;
            const int shift = i % 128 / 32;
            const int code = ((q[32 + 32 * (i / 128) + i % 32] >> (2 * shift)) & 3) -
                             ((q[i % 32] & (1 << (i / 32))) ? 0 : 4);
            values[i] = scale * ((low | (high << 4)) - 32) * code;
        }
    } else if (type == 12 || type == 13) {
        const float scale = half(bytes, 0), minimum = half(bytes, 2);
        const uint8_t* scales = q + 4;
        for (int group = 0; group < 8; ++group) {
            const int s = group < 4 ? scales[group] & 63 : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
            const int m = group < 4 ? scales[group + 4] & 63 : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
            for (int lane = 0; lane < 32; ++lane) {
                const int low = (q[(type == 12 ? 16 : 48) + 32 * (group / 2) + lane] >> (4 * (group % 2))) & 15;
                const int high = type == 13 ? (q[16 + lane] >> group) & 1 : 0;
                values[32 * group + lane] = scale * s * (low | (high << 4)) - minimum * m;
            }
        }
    } else {
        const float scale = half(bytes, 208);
        for (int i = 0; i < 256; ++i) {
            const int chunk = i / 128, part = (i / 32) % 4, lane = i % 32;
            const int low = (q[64 * chunk + 32 * (part % 2) + lane] >> (4 * (part / 2))) & 15;
            const int high = (q[128 + 32 * chunk + lane] >> (2 * part)) & 3;
            const int s = bit_value<int8_t>(q[192 + 8 * chunk + 2 * part + lane / 16]);
            values[i] = scale * s * ((low | (high << 4)) - 32);
        }
    }
}

}  // namespace

GgufFile::GgufFile(const std::filesystem::path& path) : path_(path) {
    BinaryInput input(path);
    require(input.text(4) == "GGUF", "Invalid little-endian GGUF signature");
    const uint64_t version = input.le(4), tensor_count = input.le(8), metadata_count = input.le(8);
    require(version == 2 || version == 3, "Unsupported GGUF version");
    require(metadata_count <= input.remaining() / 12, "Invalid GGUF metadata count");
    std::set<std::string> keys;
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < metadata_count; ++i) {
        const std::string key = input.text(input.le(8));
        const uint32_t type = uint32_t(input.le(4));
        require(!key.empty() && keys.insert(key).second, "Invalid or duplicate GGUF metadata key");
        if (key == "general.alignment") {
            require(type == 4, "Invalid GGUF alignment type");
            alignment = input.le(4);
            require(alignment && !(alignment & (alignment - 1)), "Invalid GGUF alignment");
        } else if (type == 8) strings_.emplace(key, input.text(input.le(8)));
        else if (type == 9) skip_array(input);
        else numbers_.emplace(key, number(input, type));
    }
    require(tensor_count && tensor_count <= input.remaining() / 32, "Invalid GGUF tensor count");
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const std::string name = input.text(input.le(8));
        const uint64_t rank = input.le(4);
        require(!name.empty() && rank >= 1 && rank <= 4, "Invalid GGUF tensor descriptor");
        Shape shape(static_cast<size_t>(rank));
        for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
            const uint64_t dimension = input.le(8);
            require(dimension && dimension <= uint64_t(std::numeric_limits<int>::max()), "Invalid GGUF dimension");
            *it = int(dimension);
        }
        const uint32_t type = uint32_t(input.le(4));
        const uint64_t offset = input.le(8);
        const Encoding format = encoding(type);
        const uint64_t count = shape_element_count(shape);
        require(shape.back() % format.block == 0 && offset % alignment == 0 &&
                count / format.block <= (std::numeric_limits<uint64_t>::max() - offset) / format.bytes,
                "Invalid GGUF tensor alignment or extent");
        const uint64_t storage = count / format.block * format.bytes;
        require(tensors_.emplace(name, SafetensorInfo{format.dtype, shape, offset, storage}).second,
                "Duplicate GGUF tensor " + name);
        types_.emplace(name, type);
        ranges.emplace_back(offset, offset + storage);
    }
    const uint64_t padding = (alignment - input.position() % alignment) % alignment;
    input.skip(padding);
    data_start_ = input.position();
    std::sort(ranges.begin(), ranges.end());
    uint64_t end = 0;
    for (const std::pair<uint64_t, uint64_t>& range : ranges) {
        require(range.first >= end && range.second <= input.remaining(), "Overlapping or truncated GGUF tensor data");
        end = range.second;
    }
}

std::vector<uint8_t> GgufFile::read(const std::string& name) const {
    const SafetensorInfo& info = tensors_.at(name);
    const unsigned int type = types_.at(name);
    const Encoding format = encoding(type);
    BinaryInput input(path_);
    input.seek(data_start_ + info.offset);
    if (format.block == 1) return input.bytes(info.bytes);
    std::vector<uint8_t> result(dtype_storage_bytes(info.dtype, shape_element_count(info.shape)));
    std::vector<uint8_t> compressed(format.bytes * 1024);
    float values[256];
    for (uint64_t offset = 0; offset < info.bytes;) {
        const size_t count = size_t(std::min<uint64_t>(1024, (info.bytes - offset) / format.bytes));
        input.read(compressed.data(), count * format.bytes);
        for (size_t i = 0; i < count; ++i) {
            decode_block(type, {compressed.data() + i * format.bytes, format.bytes}, values);
            std::memcpy(result.data() + (offset / format.bytes + i) * format.block * sizeof(float),
                        values, format.block * sizeof(float));
        }
        offset += count * format.bytes;
    }
    return result;
}

Tensor GgufFile::load(const std::string& name, Backend backend) const {
    const SafetensorInfo& info = tensors_.at(name);
    return Tensor::from_raw(info.shape, info.dtype, read(name), backend);
}

}  // namespace elman
