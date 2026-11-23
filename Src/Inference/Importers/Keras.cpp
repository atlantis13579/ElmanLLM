#include "Inference/Importers/Keras.h"
#include "Core/LocalFunction.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <tuple>

#include "Core/Device.h"
#include "IO/Hdf5.h"
#include "IO/TensorArchive.h"
#include "IO/ZipArchive.h"
#include "Inference/GraphBuilder.h"

namespace elman {
namespace {

using io::require;

struct Edge {
    std::string layer;
    int call, output;
    bool operator<(const Edge& other) const {
        return std::tie(layer, call, output) < std::tie(other.layer, other.call, other.output);
    }
};

bool is_edge(const Json& value) {
    return value.is_array() && value.array().size() >= 3 && !value.at(0).is_array() && !value.at(0).is_object();
}

Edge edge(const Json& value) {
    Edge result{value.at(0).string(), value.at(1).integer(), value.at(2).integer()};
    require(!result.layer.empty() && result.call >= 0 && result.output >= 0, "Invalid Keras graph edge");
    return result;
}

void collect_edges(const Json& data, std::vector<Edge>& out) {
    if (data.is_object()) {
        if (data.string_or("class_name", "") == "__keras_tensor__")
            out.push_back(edge(data.at("config").at("keras_history")));
        else for (const std::pair<const std::string, Json>& item : data.object()) collect_edges(item.second, out);
    } else if (is_edge(data)) out.push_back(edge(data));
    else if (data.is_array()) for (const Json& item : data.array()) collect_edges(item, out);
}

std::vector<int> integers(const Json& value) {
    std::vector<int> result;
    for (const Json& item : value.array()) result.push_back(item.is_null() ? -1 : item.integer());
    return result;
}

DType keras_dtype(const Json& config) {
    if (!config.contains("dtype") || config.at("dtype").is_null()) return DType::Float32;
    const Json& dtype = config.at("dtype");
    const std::string name = dtype.is_object() ? dtype.at("config").at("name").string() : dtype.string();
    if (name == "mixed_float16") return DType::Float16;
    if (name == "mixed_bfloat16") return DType::BFloat16;
    return parse_dtype(name);
}

TensorSpec input_spec(const Json& config) {
    for (const char* name : {"batch_shape", "batch_input_shape"})
        if (config.contains(name)) return {keras_dtype(config), integers(config.at(name))};
    if (config.contains("input_shape")) {
        std::vector<int> dims = integers(config.at("input_shape"));
        dims.insert(dims.begin(), config.int_or("batch_size", -1));
        return {keras_dtype(config), std::move(dims)};
    }
    throw std::invalid_argument("Keras input layer is missing its batch shape");
}

std::string snake_case(const std::string& type) {
    std::string result;
    for (size_t i = 0; i < type.size(); ++i) {
        const unsigned char current = static_cast<unsigned char>(type[i]);
        if (i && std::isupper(current) && std::islower(static_cast<unsigned char>(type[i - 1]))) result += '_';
        result += char(std::tolower(current));
    }
    return result;
}

class Weights {
public:
    Weights(io::Hdf5File& file, bool archive) : file_(file), archive_(archive) {
        if (!archive_) return;
        for (const std::string& child : file_.children("/layers")) {
            const std::string path = "/layers/" + child + "/vars";
            if (!file_.contains(path)) continue;
            const std::string name = file_.attribute_text(path, "name", false);
            if (!name.empty()) require(by_name_.emplace(name, path).second, "Duplicate Keras weight group " + name);
        }
    }
    Tensor load(const std::string& layer, const std::string& probable_group, const std::string& role, int index, Backend backend) {
        if (archive_) {
            const auto found = by_name_.find(layer);
            const std::string group = found == by_name_.end() ? "/layers/" + probable_group + "/vars" : found->second;
            return file_.tensor(group + '/' + std::to_string(index), backend);
        }
        const std::string group = "/model_weights/" + layer;
        if (file_.contains(group)) {
            std::vector<std::string> candidates = find(group, role, 0);
            if (candidates.empty() && role == "depthwise_kernel") candidates = find(group, "kernel", 0);
            require(candidates.size() == 1, "Missing or ambiguous Keras HDF5 weight " + layer + '/' + role);
            return file_.tensor(candidates.front(), backend);
        }
        throw std::invalid_argument("Missing Keras HDF5 weight group " + layer);
    }

private:
    std::vector<std::string> find(const std::string& path, const std::string& name, unsigned depth) {
        require(depth < 16, "Keras weight hierarchy is too deeply nested");
        std::vector<std::string> result;
        for (const std::string& child : file_.children(path)) {
            const std::string entry = path + '/' + child;
            if (child == name || child == name + ":0") result.push_back(entry);
            else {
                std::vector<std::string> matches = find(entry, name, depth + 1);
                result.insert(result.end(), matches.begin(), matches.end());
            }
        }
        return result;
    }

    io::Hdf5File& file_;
    bool archive_;
    std::map<std::string, std::string> by_name_;
};

struct Layer {
    std::string name, type, group;
    Json config;
    std::vector<std::vector<Edge>> calls;
};

class Lowering {
public:
    Lowering(GraphBuilder& graph, Weights& weights) : graph_(graph), weights_(weights) {}

    void model(const Json& model) {
        const std::string type = model.at("class_name").string();
        require(type == "Functional" || type == "Sequential" || type == "Model", "Unsupported Keras model class " + type);
        const Json& config = model.at("config");
        const bool sequential = type == "Sequential";
        const Json::Array& definitions = config.is_array() ? config.array() : config.at("layers").array();
        std::vector<std::string> order;
        std::map<std::string, int> class_counts;
        for (const Json& definition : definitions) {
            const std::string kind = definition.at("class_name").string();
            const Json& options = definition.at("config");
            const std::string name = options.at("name").string();
            std::string group = snake_case(kind);
            int& count = class_counts[group];
            if (count) group += '_' + std::to_string(count);
            ++count;
            Layer layer{name, kind, std::move(group), options, {}};
            if (sequential && kind != "InputLayer") {
                if (order.empty()) {
                    const TensorSpec shape = config.is_object() && config.contains("build_input_shape") ?
                        TensorSpec{keras_dtype(options), integers(config.at("build_input_shape"))} : input_spec(options);
                    add_input({"input", 0, 0}, shape);
                }
                layer.calls = {{{order.empty() ? "input" : order.back(), 0, 0}}};
            } else if (kind != "InputLayer") {
                for (const Json& call : definition.at("inbound_nodes").array()) {
                    std::vector<Edge> inputs;
                    if (call.is_object()) {
                        if (call.contains("kwargs") && call.at("kwargs").bool_or("training", false))
                            throw std::invalid_argument("Keras training graph execution is unsupported");
                        collect_edges(call.at("args"), inputs);
                    } else collect_edges(call, inputs);
                    layer.calls.push_back(std::move(inputs));
                }
            }
            require(layers_.emplace(name, std::move(layer)).second, "Duplicate Keras layer " + name);
            order.push_back(name);
        }
        require(!order.empty(), "Keras model has no layers");
        if (sequential) {
            const Layer& first = layers_.at(order.front());
            if (first.type == "InputLayer") add_input({first.name, 0, 0}, input_spec(first.config));
            add_output(order.back(), {order.back(), 0, 0});
            return;
        }
        std::vector<Edge> inputs;
        collect_edges(config.at("input_layers"), inputs);
        for (const Edge& input : inputs) {
            const Layer& layer = layers_.at(input.layer);
            require(layer.type == "InputLayer" && !input.call && !input.output, "Unsupported Keras symbolic input");
            add_input(input, input_spec(layer.config));
        }
        const Json& outputs = config.at("output_layers");
        if (outputs.is_object()) {
            for (const std::pair<const std::string, Json>& entry : outputs.object()) {
                const std::string& key = entry.first;
                const Json& reference = entry.second;
                std::vector<Edge> values;
                collect_edges(reference, values);
                require(values.size() == 1, "Nested Keras output containers are unsupported");
                add_output(key, values.front());
            }
        } else {
            std::vector<Edge> values;
            collect_edges(outputs, values);
            std::set<std::string> names;
            for (const Edge& item : values) {
                std::string name = item.layer;
                if (!names.insert(name).second) {
                    name += '_' + std::to_string(item.call) + '_' + std::to_string(item.output);
                    require(names.insert(name).second, "Duplicate Keras output port");
                }
                add_output(name, item);
            }
        }
    }

private:
    void add_input(const Edge& ref, const TensorSpec& spec) {
        require(values_.emplace(ref, graph_.input(ref.layer, spec)).second, "Duplicate Keras input edge");
    }
    void add_output(const std::string& name, const Edge& ref) {
        graph_.output(name, lower(ref, 0), TensorSpec{DType::Float32, {}, true, true});
    }
    std::string lower(const Edge& ref, unsigned depth) {
        require(depth < 512, "Keras graph exceeds the importer nesting limit");
        if (const auto found = values_.find(ref); found != values_.end()) return found->second;
        require(!ref.output && active_.insert(ref).second, "Keras layer has an unsupported output index or cycle");
        const Layer& layer = layers_.at(ref.layer);
        std::vector<std::string> inputs;
        for (const Edge& input : layer.calls.at(ref.call)) inputs.push_back(lower(input, depth + 1));
        std::string result;
        try { result = operation(layer, inputs); }
        catch (const std::exception& error) { throw std::invalid_argument(layer.name + " (" + layer.type + "): " + error.what()); }
        active_.erase(ref);
        values_.emplace(ref, result);
        return result;
    }

    std::string weight(const Layer& layer, const std::string& name, int index) {
        const std::pair<std::string, std::string> key = std::make_pair(layer.name, name);
        if (const auto found = constants_.find(key); found != constants_.end()) return found->second;
        Tensor tensor = weights_.load(layer.name, layer.group, name, index, graph_.backend());
        const DType dtype = keras_dtype(layer.config);
        if (tensor.dtype() != dtype && is_arithmetic_float(tensor.dtype()) && is_arithmetic_float(dtype)) tensor = cast(tensor, dtype);
        std::string value = graph_.constant(std::move(tensor));
        constants_.emplace(key, value);
        return value;
    }

    std::string activation(const std::string& type, const std::string& value) {
        if (type == "linear") return value;
        if (type == "relu6") return graph_.operation("Clip", {value}, {{"min", 0}, {"max", 6}});
        static const std::map<std::string, std::string> names{
            {"relu", "Relu"}, {"sigmoid", "Sigmoid"}, {"tanh", "Tanh"}, {"softmax", "Softmax"},
            {"log_softmax", "LogSoftmax"}, {"silu", "Silu"}, {"swish", "Silu"},
            {"gelu", "ExactGelu"}, {"exp", "Exp"}, {"exponential", "Exp"},
        };
        const auto found = names.find(type);
        require(found != names.end(), "Unsupported Keras activation " + type);
        return graph_.operation(found->second, {value});
    }

    bool channels_first(const Json& config) const {
        const std::string layout = config.string_or("data_format", "channels_last");
        require(layout == "channels_first" || layout == "channels_last", "Unsupported Keras channel layout");
        return layout == "channels_first";
    }

    std::string spatial(const Layer& layer, const std::string& input) {
        const std::string& type = layer.type;
        const Json& config = layer.config;
        const bool first = channels_first(config), pool = type == "MaxPooling2D" || type == "AveragePooling2D";
        const std::string padding = config.string_or("padding", "valid");
        const std::vector<int> kernel = pool ? integers(config.at("pool_size")) : std::vector<int>{1, 1};
        const std::vector<int> stride = !config.contains("strides") || config.at("strides").is_null() ? kernel : integers(config.at("strides"));
        require(stride.size() == 2 && stride[0] > 0 && stride[1] > 0 && (padding == "same" || padding == "valid"),
                "Invalid Keras spatial geometry");
        Json::Object attrs{{"stride_h", stride[0]}, {"stride_w", stride[1]}, {"same", padding == "same"}};
        std::string image = first ? graph_.operation("Transpose", {input}, {{"perm", Json::Array{0, 2, 3, 1}}}) : input;
        if (pool) {
            require(kernel.size() == 2 && kernel[0] > 0 && kernel[0] == kernel[1], "Keras pooling requires a square kernel");
            attrs.emplace("maximum", type == "MaxPooling2D");
            attrs.emplace("kernel", kernel[0]);
            image = graph_.operation("NhwcPool", {image}, std::move(attrs));
        } else {
            require(!config.contains("dilation_rate") || integers(config.at("dilation_rate")) == std::vector<int>{1, 1},
                    "Keras convolution requires unit dilation");
            const bool depthwise = type != "Conv2D", separable = type == "SeparableConv2D";
            std::string filter = weight(layer, depthwise ? "depthwise_kernel" : "kernel", 0);
            filter = depthwise ? graph_.operation("DepthwiseFilter", {filter}) :
                         graph_.operation("Transpose", {filter}, {{"perm", Json::Array{3, 0, 1, 2}}});
            attrs.emplace("depthwise", depthwise);
            attrs.emplace("group", config.int_or("groups", 1));
            image = graph_.operation("NhwcConv", {image, filter}, std::move(attrs));
            if (separable) {
                filter = graph_.operation("Transpose", {weight(layer, "pointwise_kernel", 1)}, {{"perm", Json::Array{3, 0, 1, 2}}});
                image = graph_.operation("NhwcConv", {image, filter}, {{"stride_h", 1}, {"stride_w", 1}});
            }
            if (config.bool_or("use_bias", true)) image = graph_.operation("Add", {image, weight(layer, "bias", separable ? 2 : 1)});
            image = activation(config.string_or("activation", "linear"), image);
        }
        return first ? graph_.operation("Transpose", {image}, {{"perm", Json::Array{0, 3, 1, 2}}}) : image;
    }

    std::string operation(const Layer& layer, const std::vector<std::string>& inputs) {
        const Json& config = layer.config;
        const std::string& type = layer.type;
        require(!inputs.empty(), "Missing Keras layer input");
        std::string value = inputs.front();
        static const std::map<std::string, std::string> merges{
            {"Add", "Add"}, {"Multiply", "Mul"}, {"Subtract", "Sub"}, {"Average", "Add"},
            {"Maximum", "Maximum"}, {"Minimum", "Minimum"},
        };
        if (const auto merge = merges.find(type); merge != merges.end()) {
            require(inputs.size() >= 2 && (type != "Subtract" || inputs.size() == 2),
                    "Invalid Keras merge input count");
            for (size_t i = 1; i < inputs.size(); ++i) value = graph_.operation(merge->second, {value, inputs[i]});
            return type == "Average" ? graph_.operation("ScalarBinary", {value},
                       {{"op", "Div"}, {"value", double(inputs.size())}}) : value;
        }
        if (type == "Concatenate") return graph_.operation("Concat", inputs, {{"axis", config.int_or("axis", -1)}});
        require(inputs.size() == 1, "Keras layer requires one input");
        if (type == "Dense") {
            require(!config.contains("quantization_config") || config.at("quantization_config").is_null(),
                    "Quantized Keras dense layers are unsupported");
            value = graph_.operation("MatMul", {value, weight(layer, "kernel", 0)});
            if (config.bool_or("use_bias", true)) value = graph_.operation("Add", {value, weight(layer, "bias", 1)});
            return activation(config.string_or("activation", "linear"), value);
        }
        if (type == "Conv2D" || type == "DepthwiseConv2D" || type == "SeparableConv2D" ||
            type == "MaxPooling2D" || type == "AveragePooling2D") return spatial(layer, value);
        if (type == "GlobalAveragePooling2D" || type == "GlobalMaxPooling2D") {
            const bool first = channels_first(config);
            const std::string dims = graph_.constant(Tensor::from_int_data({2}, {first ? 2 : 1, first ? 3 : 2}, graph_.backend()));
            return graph_.operation(type == "GlobalAveragePooling2D" ? "Mean" : "ReduceMax",
                                    {value, dims}, {{"keep_dims", config.bool_or("keepdims", false)}});
        }
        if (type == "Activation") return activation(config.at("activation").string(), value);
        if (type == "ReLU") {
            require(config.number_or("negative_slope", 0) == 0 && config.number_or("threshold", 0) == 0,
                    "Unsupported Keras ReLU slope or threshold");
            if (config.contains("max_value") && !config.at("max_value").is_null())
                return graph_.operation("Clip", {value}, {{"min", 0}, {"max", config.at("max_value")}});
            return graph_.operation("Relu", {value});
        }
        if (type == "LeakyReLU")
            return graph_.operation("LeakyRelu", {value}, {{"alpha", config.number_or("negative_slope", config.number_or("alpha", 0.3))}});
        if (type == "Softmax") {
            const Json& axis = config.at("axis");
            require(!axis.is_array() || axis.array().size() == 1, "Keras Softmax requires a single axis");
            return graph_.operation(type, {value}, {{"axis", axis.is_array() ? axis.at(0) : axis}});
        }
        if (type == "Dropout" || type == "SpatialDropout1D" || type == "SpatialDropout2D" || type == "GaussianNoise")
            return value;
        if (type == "Flatten") {
            if (channels_first(config)) value = graph_.operation("ChannelsLast", {value});
            return graph_.operation("FlattenRange", {value}, {{"start", 1}});
        }
        if (type == "ZeroPadding2D") {
            const Json& padding = config.at("padding");
            require(padding.is_array() && padding.array().size() == 2, "Invalid Keras zero padding");
            LocalFunction pair{[&](size_t index) {
                const Json& item = padding.at(index);
                std::vector<int> result = item.is_array() ? integers(item) : std::vector<int>{item.integer(), item.integer()};
                require(result.size() == 2, "Invalid Keras zero padding pair");
                return result;
            }};
            const std::vector<int> h = pair(0), w = pair(1);
            const Json::Array pads = channels_first(config) ? Json::Array{0, 0, 0, 0, h[0], h[1], w[0], w[1]} :
                                                     Json::Array{0, 0, h[0], h[1], w[0], w[1], 0, 0};
            return graph_.operation("Pad", {value}, {{"pads", pads}});
        }
        if (type == "Reshape") return graph_.operation("BatchReshape", {value}, {{"shape", config.at("target_shape")}});
        if (type == "Permute") {
            Json::Array dims = config.at("dims").array();
            dims.insert(dims.begin(), 0);
            return graph_.operation("Transpose", {value}, {{"perm", std::move(dims)}});
        }
        if (type == "Embedding") return graph_.operation("Gather", {weight(layer, "embeddings", 0), value});
        if (type == "BatchNormalization") {
            int index = 0;
            const std::string scale = config.bool_or("scale", true) ? weight(layer, "gamma", index++) : "";
            const std::string bias = config.bool_or("center", true) ? weight(layer, "beta", index++) : "";
            const std::string mean = weight(layer, "moving_mean", index++);
            const std::string variance = weight(layer, "moving_variance", index);
            const Json& axis = config.at("axis");
            require(!axis.is_array() || axis.array().size() == 1, "Keras batch normalization requires one axis");
            return graph_.operation("AffineBatchNormalization", {value, mean, variance, scale, bias},
                                   {{"axis", axis.is_array() ? axis.at(0) : axis}, {"epsilon", config.number_or("epsilon", 1e-3)}});
        }
        if (type == "LayerNormalization") {
            const Json& axis = config.at("axis");
            require((axis.is_array() ? integers(axis) : std::vector<int>{axis.integer()}) == std::vector<int>{-1},
                    "Keras LayerNormalization requires the last axis");
            const bool scale = config.bool_or("scale", true), center = config.bool_or("center", true);
            return graph_.operation("LayerNormalization", {value, scale ? weight(layer, "gamma", 0) : "",
                                    center ? weight(layer, "beta", scale ? 1 : 0) : ""},
                                    {{"epsilon", config.number_or("epsilon", 1e-3)}});
        }
        throw std::invalid_argument("Unsupported Keras layer");
    }

    GraphBuilder& graph_;
    Weights& weights_;
    std::map<std::string, Layer> layers_;
    std::map<Edge, std::string> values_;
    std::set<Edge> active_;
    std::map<std::pair<std::string, std::string>, std::string> constants_;
};

InferenceSession compile(const Json& config, io::Hdf5File& file, bool archive, Backend backend, int device) {
    Weights weights(file, archive);
    GraphBuilder builder(backend, device);
    Lowering(builder, weights).model(config);
    return std::move(builder).finish();
}

}  // namespace

InferenceSession load_keras_session(const std::filesystem::path& path, Backend backend, int device) {
    DeviceScope owner(backend, device);
    if (model_file_extension(path) == ".keras") {
        io::ZipArchive archive(path);
        const std::vector<uint8_t> config = archive.read("config.json");
        io::Hdf5File file(archive.read("model.weights.h5"));
        return compile(Json::parse(io::view(config).text(0, config.size())), file, true, backend, device);
    }
    io::Hdf5File file(path);
    return compile(Json::parse(file.attribute_text("/", "model_config")), file, false, backend, device);
}

}  // namespace elman
