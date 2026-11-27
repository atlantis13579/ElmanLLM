#include "ToolIO.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>

#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>

#include "Core/TensorDispatch.h"
#include "IO/AtomicFile.h"
#include "IO/Safetensors.h"

namespace {
namespace onnx = ::ONNX_NAMESPACE;
namespace fs = std::filesystem;
using namespace elman;

DType dtype(int type) {
    using T = onnx::TensorProto;
    switch (type) {
    case T::BOOL: return DType::Bool;
    case T::FLOAT: return DType::Float32;
    case T::FLOAT16: return DType::Float16;
    case T::DOUBLE: return DType::Float64;
    case T::INT8: return DType::Int8;
    case T::INT16: return DType::Int16;
    case T::INT32: return DType::Int32;
    case T::INT64: return DType::Int64;
    case T::UINT8: return DType::UInt8;
    case T::UINT16: return DType::UInt16;
    case T::UINT32: return DType::UInt32;
    case T::UINT64: return DType::UInt64;
    default: throw std::invalid_argument("Unsupported ONNX tensor type " + std::to_string(type));
    }
}

int dimension(int64_t value) {
    if (value < 0 || value > std::numeric_limits<int>::max())
        throw std::invalid_argument("ONNX dimension exceeds the Elman indexing limit");
    return static_cast<int>(value);
}

struct TensorLayout {
    Shape shape;
    DType type;
    size_t bytes;
};

TensorLayout layout(const onnx::TensorProto& tensor) {
    if (tensor.dims_size() > 8) throw std::invalid_argument("ONNX tensor rank exceeds eight");
    Shape shape;
    for (int64_t size : tensor.dims()) shape.push_back(dimension(size));
    const DType type = dtype(tensor.data_type());
    const size_t bytes = dtype_storage_bytes(type, shape_element_count(shape));
    return {std::move(shape), type, bytes};
}

Json port(const onnx::ValueInfoProto& value) {
    if (!value.type().has_tensor_type() || !value.type().tensor_type().has_shape() ||
        value.type().tensor_type().shape().dim_size() > 8)
        throw std::invalid_argument("Expected a tensor port with known rank of at most eight: " + value.name());
    const onnx::TypeProto_Tensor& tensor = value.type().tensor_type();
    Json::Array shape;
    for (const onnx::TensorShapeProto_Dimension& dim : tensor.shape().dim()) shape.emplace_back(dim.has_dim_value() ? dimension(dim.dim_value()) : -1);
    return Json::Object{{"dtype", dtype_name(dtype(tensor.elem_type()))}, {"shape", std::move(shape)}};
}

uint64_t external_integer(const std::string& text) {
    uint64_t value = 0;
    const char* const end = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) throw std::invalid_argument("Invalid ONNX external data offset or length");
    return value;
}

void inline_external_data(onnx::TensorProto& tensor, const fs::path& root) {
    if (tensor.data_location() != onnx::TensorProto::EXTERNAL) return;
    const TensorLayout description = layout(tensor);
    std::map<std::string, std::string> metadata;
    for (const onnx::StringStringEntryProto& item : tensor.external_data())
        if (!metadata.emplace(item.key(), item.value()).second) throw std::invalid_argument("Duplicate ONNX external data field");
    const auto location = metadata.find("location");
    if (location == metadata.end() || location->second.empty() || location->second.find('\0') != std::string::npos)
        throw std::invalid_argument("Missing ONNX external data location");
    const std::filesystem::path relative = fs::u8path(location->second);
    if (relative.has_root_path() || std::find(relative.begin(), relative.end(), fs::path("..")) != relative.end())
        throw std::invalid_argument("ONNX external data must be inside the model directory");
    const std::filesystem::path path = fs::canonical(root / relative);
    const std::filesystem::path contained = path.lexically_relative(root);
    if (contained.empty() || *contained.begin() == fs::path("..") || !fs::is_regular_file(path))
        throw std::invalid_argument("ONNX external data resolves outside the model directory or is not a file");
    const uint64_t offset = metadata.count("offset") ? external_integer(metadata.at("offset")) : 0;
    const uint64_t length = metadata.count("length") ? external_integer(metadata.at("length")) : description.bytes;
    const uintmax_t size = fs::file_size(path);
    if (length != description.bytes || offset > size || length > size - offset ||
        offset > uint64_t(std::numeric_limits<std::streamoff>::max()) || length > uint64_t(std::numeric_limits<std::streamsize>::max()))
        throw std::invalid_argument("ONNX external data range does not match the tensor: " + tensor.name());
    std::ifstream file(path, std::ios::binary);
    std::string data(static_cast<size_t>(length), '\0');
    file.seekg(static_cast<std::streamoff>(offset));
    if (!data.empty()) file.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file) throw std::runtime_error("Cannot read ONNX external tensor: " + tensor.name());
    tensor.clear_data_location();
    tensor.clear_external_data();
    tensor.set_raw_data(std::move(data));
}

template<class T>
void append_bytes(std::vector<uint8_t>& output, T value, size_t width) {
    uint64_t bits;
    if constexpr (std::is_floating_point_v<T>) {
        std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t> word;
        std::memcpy(&word, &value, sizeof(word));
        bits = word;
    } else {
        bits = static_cast<uint64_t>(value);
    }
    for (size_t i = 0; i < width; ++i) output.push_back(static_cast<uint8_t>(bits >> (i * 8)));
}

template<class Repeated>
void append_values(std::vector<uint8_t>& output, const Repeated& values, size_t width, size_t expected) {
    if (static_cast<size_t>(values.size()) != expected / width) throw std::invalid_argument("ONNX tensor payload size differs from its shape");
    output.reserve(expected);
    for (typename Repeated::value_type value : values) append_bytes(output, value, width);
}

Tensor tensor_value(const onnx::TensorProto& tensor) {
    TensorLayout description = layout(tensor);
    std::vector<uint8_t> bytes;
    if (tensor.has_raw_data()) {
        const std::string& raw = tensor.raw_data();
        if (raw.size() != description.bytes) throw std::invalid_argument("ONNX raw tensor size differs from its shape");
        bytes.assign(raw.begin(), raw.end());
    } else {
        const unsigned int width = dtype_info(description.type).bits / 8;
        switch (description.type) {
        case DType::Float32: append_values(bytes, tensor.float_data(), width, description.bytes); break;
        case DType::Float64: append_values(bytes, tensor.double_data(), width, description.bytes); break;
        case DType::Int64: append_values(bytes, tensor.int64_data(), width, description.bytes); break;
        case DType::UInt32:
        case DType::UInt64: append_values(bytes, tensor.uint64_data(), width, description.bytes); break;
        default: append_values(bytes, tensor.int32_data(), width, description.bytes); break;
        }
    }
    return Tensor::from_raw(std::move(description.shape), description.type, bytes);
}

const std::map<std::string, std::set<std::string>>& supported() {
    static const std::map<std::string, std::set<std::string>> attributes{
        {"Identity", {}}, {"Relu", {}}, {"Exp", {}}, {"Sqrt", {}}, {"Sigmoid", {}},
        {"Add", {}}, {"Sub", {}}, {"Mul", {}}, {"Div", {}}, {"MatMul", {}},
        {"Gemm", {"transA", "transB", "alpha", "beta"}}, {"Flatten", {"axis"}},
        {"Reshape", {"allowzero"}}, {"Transpose", {"perm"}}, {"Gather", {"axis"}},
        {"Concat", {"axis"}}, {"Softmax", {"axis"}}, {"Shape", {"start", "end"}}, {"Cast", {"to"}},
        {"Conv", {"strides", "pads", "dilations", "auto_pad", "kernel_shape", "group"}},
        {"MaxPool", {"strides", "pads", "dilations", "auto_pad", "kernel_shape", "ceil_mode"}},
        {"AveragePool", {"strides", "pads", "dilations", "auto_pad", "kernel_shape", "ceil_mode", "count_include_pad"}},
        {"GlobalAveragePool", {}}, {"BatchNormalization", {"epsilon", "training_mode", "momentum"}},
        {"Constant", {"value", "value_int", "value_ints", "value_float", "value_floats"}},
    };
    return attributes;
}

double exact_integer(int64_t value) {
    if (value < -9007199254740991ll || value > 9007199254740991ll)
        throw std::invalid_argument("ONNX integer attribute exceeds the supported JSON range");
    return double(value);
}

Json attribute_value(const onnx::AttributeProto& attr) {
    using A = onnx::AttributeProto;
    switch (attr.type()) {
    case A::FLOAT:
        if (!std::isfinite(attr.f())) throw std::invalid_argument("Nonfinite ONNX attribute " + attr.name());
        return double(attr.f());
    case A::INT: return exact_integer(attr.i());
    case A::STRING:
        if (!std::all_of(attr.s().begin(), attr.s().end(), [](unsigned char c) { return c >= 32 && c < 127; }))
            throw std::invalid_argument("Invalid ONNX string attribute " + attr.name());
        return attr.s();
    case A::INTS: {
        Json::Array values;
        for (const int64_t value : attr.ints()) values.emplace_back(exact_integer(value));
        return values;
    }
    default: throw std::invalid_argument("Unsupported ONNX attribute type for " + attr.name());
    }
}

Tensor constant(const onnx::AttributeProto& attr) {
    if (attr.name() == "value") return tensor_value(attr.t());
    using A = onnx::AttributeProto;
    std::vector<uint8_t> data;
    switch (attr.type()) {
    case A::INT:
        append_bytes(data, attr.i(), 8);
        return Tensor::from_raw({}, DType::Int64, data);
    case A::FLOAT:
        append_bytes(data, attr.f(), 4);
        return Tensor::from_raw({}, DType::Float32, data);
    case A::INTS:
        append_values(data, attr.ints(), 8, size_t(attr.ints_size()) * 8);
        return Tensor::from_raw({attr.ints_size()}, DType::Int64, data);
    case A::FLOATS:
        append_values(data, attr.floats(), 4, size_t(attr.floats_size()) * 4);
        return Tensor::from_raw({attr.floats_size()}, DType::Float32, data);
    default: throw std::invalid_argument("Unsupported ONNX Constant value");
    }
}

struct Bundle {
    Json::Object manifest;
    std::map<std::string, Tensor> weights;
};

Bundle import_graph(const fs::path& source) {
    std::ifstream input(source, std::ios::binary);
    onnx::ModelProto model;
    if (!input || !model.ParseFromIstream(&input) || !model.has_graph())
        throw std::invalid_argument("Cannot read ONNX model: " + source.u8string());
    if (model.functions_size() || model.training_info_size() || model.graph().sparse_initializer_size() ||
        model.opset_import_size() != 1)
        throw std::invalid_argument("ONNX import supports one standard opset without local functions, training graphs or sparse initializers");
    onnx::OperatorSetIdProto& opset = *model.mutable_opset_import(0);
    if ((opset.domain() != "" && opset.domain() != "ai.onnx") || opset.version() < 13)
        throw std::invalid_argument("ONNX standard opset 13 or later is required");
    opset.clear_domain();
    const std::filesystem::path root = fs::canonical(fs::absolute(source).parent_path());
    for (onnx::TensorProto& tensor : *model.mutable_graph()->mutable_initializer()) inline_external_data(tensor, root);
    for (onnx::NodeProto& node : *model.mutable_graph()->mutable_node()) {
        const auto op = supported().find(node.op_type());
        const std::string name = node.name().empty() ? node.op_type() : node.name();
        if (op == supported().end()) throw std::invalid_argument(name + ": unsupported ONNX kernal " + node.op_type());
        if ((node.domain() != "" && node.domain() != "ai.onnx") ||
            std::count_if(node.output().begin(), node.output().end(), [](const std::string& s) { return !s.empty(); }) != 1)
            throw std::invalid_argument(name + ": expected one named output in the standard ONNX domain");
        node.clear_domain();
        for (onnx::AttributeProto& attr : *node.mutable_attribute()) {
            if (!op->second.count(attr.name())) throw std::invalid_argument(name + ": unsupported attribute " + attr.name());
            if (node.op_type() == "Constant" && attr.name() == "value") inline_external_data(*attr.mutable_t(), root);
        }
    }
    onnx::checker::check_model(model);
    onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(), onnx::ShapeInferenceOptions(true, 1));

    Bundle result;
    const onnx::GraphProto& graph = model.graph();
    for (const onnx::TensorProto& tensor : graph.initializer()) result.weights.emplace(tensor.name(), tensor_value(tensor));
    Json::Object inputs, outputs;
    for (const onnx::ValueInfoProto& value : graph.input())
        if (!result.weights.count(value.name())) inputs.emplace(value.name(), port(value));
    for (const onnx::ValueInfoProto& value : graph.output()) outputs.emplace(value.name(), port(value));
    Json::Array nodes;
    for (const onnx::NodeProto& node : graph.node()) {
        const std::string& output = *std::find_if(node.output().begin(), node.output().end(), [](const std::string& s) { return !s.empty(); });
        try {
            if (node.op_type() == "Constant") {
                if (node.attribute_size() != 1) throw std::invalid_argument("Constant needs one value attribute");
                if (!result.weights.emplace(output, constant(node.attribute(0))).second)
                    throw std::invalid_argument("Duplicate output initializer");
                continue;
            }
            Json::Object attributes;
            for (const onnx::AttributeProto& attr : node.attribute()) {
                attributes.emplace(attr.name(), node.op_type() == "Cast" && attr.name() == "to"
                    ? Json(dtype_name(dtype(dimension(attr.i())))) : attribute_value(attr));
            }
            nodes.emplace_back(Json::Object{{"op", node.op_type()}, {"name", node.name().empty() ? output : node.name()},
                {"inputs", Json::Array(node.input().begin(), node.input().end())}, {"output", output},
                {"attributes", std::move(attributes)}});
        } catch (const std::exception& error) {
            throw std::invalid_argument((node.name().empty() ? output : node.name()) + ": " + error.what());
        }
    }
    result.manifest = {{"format", "elman.graph"}, {"version", 1}, {"inputs", std::move(inputs)},
                       {"outputs", std::move(outputs)}, {"nodes", std::move(nodes)}};
    return result;
}

void write_bundle(const fs::path& destination, Bundle bundle) {
    write_file_atomically(destination, [&](const fs::path& temporary) {
        if (!bundle.weights.empty()) {
            struct StagedWeights {
                fs::path path;
                ~StagedWeights() { std::error_code ignored; fs::remove(path, ignored); }
            } staged{temporary};
            staged.path += ".safetensors";
            save_safetensors(staged.path, bundle.weights);
            const std::string digest = tools::sha256_file(staged.path);
            const std::string name = destination.stem().u8string() + '.' + digest.substr(0, 16) + ".safetensors";
            const fs::path weights = destination.parent_path() / fs::u8path(name);
            bool published = false;
            if (!fs::exists(weights)) {
                std::error_code error;
                fs::rename(staged.path, weights, error);
                if (error && !fs::is_regular_file(weights)) throw fs::filesystem_error("Cannot publish ONNX weights", weights, error);
                published = !error;
            }
            if (!published && tools::sha256_file(weights) != digest) throw std::runtime_error("ONNX initializer name is already occupied");
            bundle.manifest["initializers"] = name;
        }
        std::ofstream manifest(temporary, std::ios::binary);
        tools::write_json(manifest, bundle.manifest);
        manifest.close();
        if (!manifest) throw std::runtime_error("Cannot close ONNX graph manifest");
    });
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc == 2 && (fs::path(argv[1]) == "--help" || fs::path(argv[1]) == "-h")) {
            std::cout << "Usage: elman_import_onnx MODEL.onnx OUTPUT.json\n";
            return 0;
        }
        if (argc != 3) throw std::invalid_argument("Expected an ONNX source and Elman graph destination");
        write_bundle(argv[2], import_graph(argv[1]));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Elman ONNX import: " << error.what() << '\n';
        return 1;
    }
}
