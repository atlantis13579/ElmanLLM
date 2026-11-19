#include "Inference/Importers/TfLite.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

#include "Core/Device.h"
#include "Core/TensorDispatch.h"
#include "IO/BinaryInput.h"
#include "IO/Json.h"
#include "Inference/GraphSession.h"

namespace elman {
namespace {

using io::Bytes;
using io::require;

struct Vector {
    Bytes bytes;
    size_t count = 0, width = 0;
    uint64_t raw(size_t index) const {
        require(index < count, "TFLite vector index is out of range");
        return bytes.le(index * width, width);
    }
    int integer(size_t index) const {
        const auto bits = uint32_t(raw(index));
        int32_t value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    std::vector<int> integers() const {
        std::vector<int> result;
        for (size_t index = 0; index < count; ++index) result.push_back(integer(index));
        return result;
    }
};

class Table {
public:
    Table(Bytes file, size_t root) : file_(file), root_(root) {
        if (!root_) return;
        const auto bits = uint32_t(file_.le(root_, 4));
        int32_t distance;
        std::memcpy(&distance, &bits, sizeof(distance));
        const auto vtable = int64_t(root_) - distance;
        require(distance && vtable >= 0 && uint64_t(vtable) < file_.size, "Invalid TFLite vtable offset");
        const size_t length = size_t(file_.le(size_t(vtable), 2));
        size_ = size_t(file_.le(size_t(vtable) + 2, 2));
        require(length >= 4 && length % 2 == 0 && size_ >= 4, "Invalid TFLite table header");
        directory_ = file_.sub(size_t(vtable) + 4, length - 4);
        file_.sub(root_, size_);
    }
    static Table entry(Bytes file, const Vector& table_offsets, size_t index) {
        const auto relative = table_offsets.raw(index);
        const auto origin = size_t(table_offsets.bytes.data - file.data) + 4 * index;
        return Table(file, target(file, origin, relative));
    }
    bool present() const { return root_ != 0; }
    bool has(size_t index) const { return field(index, 1) != 0; }
    uint64_t raw(size_t index, size_t width, uint64_t fallback = 0) const {
        const auto offset = field(index, width);
        return offset ? file_.le(offset, width) : fallback;
    }
    int integer(size_t index, int fallback = 0) const {
        const auto bits = uint32_t(raw(index, 4, uint32_t(fallback)));
        int32_t value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    float number(size_t index, float fallback) const {
        if (!has(index)) return fallback;
        const auto bits = uint32_t(raw(index, 4));
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    Table table(size_t index) const { return Table(file_, reference(index)); }
    Vector vector(size_t index, size_t width) const {
        const auto offset = reference(index);
        if (!offset) return {};
        const auto count = file_.le(offset, 4);
        require(count <= (file_.size - offset - 4) / width, "Truncated TFLite vector");
        return {file_.sub(offset + 4, size_t(count) * width), size_t(count), width};
    }
    std::string string(size_t index) const {
        const auto field = vector(index, 1);
        return field.bytes.text(0, field.count);
    }

private:
    static size_t target(Bytes bytes, size_t origin, uint64_t relative) {
        require(relative >= 4 && relative < bytes.size - origin, "Invalid TFLite offset");
        return origin + size_t(relative);
    }
    size_t field(size_t index, size_t width) const {
        if (index >= directory_.size / 2) return 0;
        const auto offset = directory_.le(index * 2, 2);
        require(!offset || (offset >= 4 && offset + width <= size_), "Invalid TFLite field offset");
        return offset ? root_ + size_t(offset) : 0;
    }
    size_t reference(size_t index) const {
        const auto offset = field(index, 4);
        return offset ? target(file_, offset, file_.le(offset, 4)) : 0;
    }

    Bytes file_, directory_;
    size_t root_, size_ = 0;
};

DType tensor_dtype(uint64_t type) {
    switch (type) {
        case 0: return DType::Float32;
        case 1: return DType::Float16;
        case 2: return DType::Int32;
        case 3: return DType::UInt8;
        case 4: return DType::Int64;
        case 6: return DType::Bool;
        case 7: return DType::Int16;
        case 8: return DType::Complex64;
        case 9: return DType::Int8;
        case 10: return DType::Float64;
        case 11: return DType::Complex128;
        case 12: return DType::UInt64;
        case 15: return DType::UInt32;
        case 16: return DType::UInt16;
        case 18: return DType::BFloat16;
        default: throw std::invalid_argument("Unsupported TFLite tensor type " + std::to_string(type));
    }
}

Json::Array array(const std::vector<int>& values) {
    return Json::Array(values.begin(), values.end());
}

struct TensorInfo {
    std::string name;
    DType dtype;
    Shape shape, signature;
    Table quantization;
};

struct Lowered {
    const char* op;
    Json::Object attributes;
    uint64_t activation = 0;
};

Lowered spatial(int code, const Table& options) {
    const bool depthwise = code == 4, pool = code == 1 || code == 17;
    const auto padding = options.raw(0, 1);
    const int sw = options.integer(1), sh = options.integer(2);
    require(padding <= 1 && sw > 0 && sh > 0, "Invalid TFLite padding or stride");
    Json::Object attributes{{"same", padding == 0}, {"stride_h", sh}, {"stride_w", sw}};
    if (pool) {
        const int width = options.integer(3), height = options.integer(4);
        require(width > 0 && width == height, "TFLite pooling requires a square filter");
        attributes.emplace("kernel", width);
        attributes.emplace("maximum", code == 17);
        return {"NhwcPool", std::move(attributes), options.raw(5, 1)};
    }
    require(options.integer(depthwise ? 5 : 4, 1) == 1 && options.integer(depthwise ? 6 : 5, 1) == 1,
            "TFLite convolution requires unit dilation");
    attributes.emplace("depthwise", depthwise);
    return {"NhwcConv", std::move(attributes), options.raw(depthwise ? 4 : 3, 1)};
}

Lowered lower(int code, const Table& options, size_t input_count, DType output_dtype) {
    switch (code) {
        case 0: return {"Add", {}, options.raw(0, 1)};
        case 1: case 3: case 4: case 17: return spatial(code, options);
        case 2: return {"Concat", {{"axis", options.integer(0)}}, options.raw(1, 1)};
        case 6: return {"Cast", {{"to", dtype_name(output_dtype)}}};
        case 9:
            require(options.raw(1, 1) == 0 && !options.raw(3, 1), "Unsupported TFLite fully connected weight format");
            return {"FullyConnected", {{"keep_dims", options.raw(2, 1) != 0}}, options.raw(0, 1)};
        case 14: return {"Sigmoid", {}};
        case 18: return {"Mul", {}, options.raw(0, 1)};
        case 19: return {"Relu", {}};
        case 20: return {"Clip", {{"min", -1}, {"max", 1}}};
        case 21: return {"Clip", {{"min", 0}, {"max", 6}}};
        case 22:
            return {"Reshape", {{"allowzero", 1}, {"shape", input_count == 1 ?
                    array(options.vector(0, 4).integers()) : Json::Array{}}}};
        case 25: return {"Softmax", {{"beta", double(options.number(0, 0))}}};
        case 28: return {"Tanh", {}};
        case 36:
            require(options.integer(1) == 0, "TFLite Gather requires batch_dims=0");
            return {"Gather", {{"axis", options.integer(0)}}};
        case 39: return {"Transpose", {}};
        case 40: return {"Mean", {{"keep_dims", options.raw(0, 1) != 0}}};
        case 41: return {"Sub", {}, options.raw(0, 1)};
        case 42: return {"Div", {}, options.raw(0, 1)};
        case 47: return {"Exp", {}};
        case 50: return {"LogSoftmax", {}};
        case 53: return {"Cast", {{"to", dtype_name(output_dtype)}}};
        case 55: return {"Maximum", {}};
        case 57: return {"Minimum", {}};
        case 59: return {"Neg", {}};
        case 73: return {"Log", {}};
        case 75: return {"Sqrt", {}};
        case 76: return {"Rsqrt", {}};
        case 92: return {"Square", {}};
        case 98: return {"LeakyRelu", {{"alpha", double(options.number(0, 0))}}};
        case 101: return {"Abs", {}};
        default: throw std::invalid_argument("Unsupported TFLite operator code " + std::to_string(code));
    }
}

Lowered activation(uint64_t code) {
    switch (code) {
        case 1: return {"Relu", {}};
        case 2: return lower(20, Table({}, 0), 1, DType::Float32);
        case 3: return lower(21, Table({}, 0), 1, DType::Float32);
        case 4: return {"Tanh", {}};
        default: throw std::invalid_argument("Unsupported TFLite fused activation " + std::to_string(code));
    }
}

void append(Json::Array& nodes, Lowered op, Json::Array inputs, const std::string& output) {
    nodes.emplace_back(Json::Object{{"op", op.op}, {"inputs", std::move(inputs)}, {"output", output},
                                    {"attributes", std::move(op.attributes)}});
}

void dequantize(Json::Array& nodes, NamedTensors& weights, const TensorInfo& x, const std::string& output,
                const std::string& temporary, Backend backend) {
    const auto scale = x.quantization.vector(2, 4), zero = x.quantization.vector(3, 8);
    require(is_standard_integer(x.dtype) && scale.count > 0 && (zero.count == 0 || zero.count == scale.count),
            "Invalid TFLite affine quantization");
    const int axis = x.quantization.integer(6);
    Shape shape;
    if (scale.count > 1) {
        require(axis >= 0 && size_t(axis) < x.shape.size() && size_t(x.shape[axis]) == scale.count,
                "Invalid TFLite per-axis quantization");
        shape.assign(x.shape.size(), 1);
        shape[axis] = int(scale.count);
    }
    std::vector<float> scales(scale.count), zeros(scale.count);
    for (size_t i = 0; i < scale.count; ++i) {
        const auto bits = uint32_t(scale.raw(i));
        std::memcpy(&scales[i], &bits, sizeof(bits));
        require(std::isfinite(scales[i]) && scales[i] > 0, "Invalid TFLite quantization scale");
        if (zero.count) {
            const auto point = zero.raw(i);
            int64_t value;
            std::memcpy(&value, &point, sizeof(value));
            zeros[i] = float(value);
        }
    }
    const auto scale_name = temporary + "/scale", zero_name = temporary + "/zero";
    weights.emplace(scale_name, Tensor::from_data(shape, scales, backend));
    weights.emplace(zero_name, Tensor::from_data(shape, zeros, backend));
    append(nodes, {"Cast", {{"to", dtype_name(DType::Float32)}}}, {x.name}, temporary);
    append(nodes, {"Sub", {}}, {temporary, zero_name}, temporary + "/centered");
    append(nodes, {"Mul", {}}, {temporary + "/centered", scale_name}, output);
}

}  // namespace

InferenceSession load_tflite_session(const std::filesystem::path& path, Backend backend, int device) {
    DeviceScope owner(backend, device);
    io::BinaryInput disk(path);
    const auto contents = disk.bytes(disk.size());
    const auto file = io::view(contents);
    require(file.text(4, 4) == "TFL3" && file.le(0, 4) >= 8, "Invalid TFLite signature");
    const Table model(file, size_t(file.le(0, 4)));
    require(model.raw(0, 4) == 3, "Unsupported TFLite schema version");
    const auto codes = model.vector(1, 4), subgraphs = model.vector(2, 4), buffers = model.vector(4, 4);
    require(subgraphs.count == 1 && buffers.count > 0, "TFLite requires a single subgraph and a buffer catalog");
    const auto graph = Table::entry(file, subgraphs, 0);
    const auto tensors = graph.vector(0, 4), operators = graph.vector(3, 4);
    const auto inputs = graph.vector(1, 4).integers(), outputs = graph.vector(2, 4).integers();
    std::set<std::string> names;
    std::vector<TensorInfo> info;
    NamedTensors initializers;
    for (size_t index = 0; index < tensors.count; ++index) {
        const auto tensor = Table::entry(file, tensors, index);
        require(!tensor.raw(5, 1) && !tensor.has(6), "TFLite sparse and variable tensors are unsupported");
        auto name = tensor.string(3);
        if (name.empty() || !names.insert(name).second) {
            name = "tensor_" + std::to_string(index);
            while (!names.insert(name).second) name += '_';
        }
        const auto shape = tensor.vector(0, 4).integers();
        const auto signature = tensor.has(7) ? tensor.vector(7, 4).integers() : shape;
        require(signature.size() == shape.size() && shape.size() <= 8, "Invalid TFLite tensor rank");
        const auto dtype = tensor_dtype(tensor.raw(1, 1));
        const auto index_buffer = size_t(tensor.raw(2, 4));
        const auto buffer = Table::entry(file, buffers, index_buffer);
        auto data = buffer.vector(0, 1).bytes;
        const auto external_offset = buffer.raw(1, 8), external_size = buffer.raw(2, 8);
        if (external_size) {
            require(!buffer.has(0) && external_offset && external_offset <= file.size &&
                    external_size <= file.size - external_offset, "Invalid external TFLite tensor buffer");
            data = file.sub(size_t(external_offset), size_t(external_size));
        }
        if (data.size || (index_buffer && buffer.has(0))) {
            require(data.size == dtype_storage_bytes(dtype, shape_element_count(shape)), "Invalid TFLite constant size");
            std::vector<uint8_t> bytes;
            if (data.size) bytes.assign(data.data, data.data + data.size);
            initializers.emplace(name, Tensor::from_raw(shape, dtype, bytes, backend));
        }
        info.push_back({std::move(name), dtype, shape, signature, tensor.table(4)});
    }
    auto signature = [&](const std::vector<int>& ports) {
        Json::Object result;
        for (int index : ports) {
            const auto& tensor = info.at(index);
            require(result.emplace(tensor.name, Json::Object{{"dtype", dtype_name(tensor.dtype)},
                    {"shape", array(tensor.signature)}}).second, "Duplicate TFLite port");
        }
        return result;
    };
    std::string prefix = "_elman";
    while (std::any_of(names.begin(), names.end(), [&](const auto& name) { return name.rfind(prefix, 0) == 0; })) prefix += '_';
    Json::Array nodes;
    for (size_t index = 0; index < operators.count; ++index) {
        const auto op = Table::entry(file, operators, index);
        require(!op.vector(5, 1).count && !op.vector(7, 1).count && !op.raw(9, 8) && !op.raw(10, 8),
                "TFLite custom or stateful operators are unsupported");
        const auto opcode = Table::entry(file, codes, size_t(op.raw(0, 4)));
        const int code = std::max(opcode.integer(3), int(opcode.raw(0, 1)));
        require(!opcode.has(1) && opcode.integer(2, 1) > 0, "Unsupported TFLite operator declaration");
        const auto arguments = op.vector(1, 4).integers(), results = op.vector(2, 4).integers();
        require(results.size() == 1, "TFLite operator must have one output");
        const auto& output = info.at(results[0]);
        require(!output.quantization.vector(2, 4).count, "TFLite quantized execution is unsupported");
        const auto temporary = prefix + '/' + std::to_string(index);
        Json::Array reads;
        for (int input : arguments) {
            if (input == -1) reads.emplace_back("");
            else {
                const auto& value = info.at(input);
                require(code == 6 || !value.quantization.vector(2, 4).count, "TFLite quantized execution is unsupported");
                reads.emplace_back(value.name);
            }
        }
        if (code == 6) {
            require(arguments.size() == 1 && arguments[0] >= 0 && output.dtype == DType::Float32,
                    "TFLite Dequantize requires one input and Float32 output");
            const auto& value = info.at(arguments[0]);
            if (value.quantization.vector(2, 4).count) {
                dequantize(nodes, initializers, value, output.name, temporary, backend);
                continue;
            }
            require(is_arithmetic_float(value.dtype), "TFLite integer Dequantize requires quantization parameters");
        }
        auto lowered = lower(code, op.table(4), reads.size(), output.dtype);
        const auto fused = lowered.activation;
        append(nodes, std::move(lowered), std::move(reads), fused ? temporary : output.name);
        if (fused) append(nodes, activation(fused), {temporary}, output.name);
    }
    return make_graph_session(Json::Object{{"inputs", signature(inputs)}, {"outputs", signature(outputs)},
                                           {"nodes", std::move(nodes)}},
                              std::move(initializers), backend, device);
}

}  // namespace elman
