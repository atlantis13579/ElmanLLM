#include "Inference/Importers/SavedModel.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <optional>
#include <set>

#include "Core/Device.h"
#include "IO/TensorFlowBundle.h"
#include "Inference/GraphBuilder.h"

namespace elman {
namespace {

using io::Bytes;
using io::protobuf::Message;
using io::require;
namespace tf = io::tensorflow;

std::vector<std::string> strings(const Message& message, uint32_t field) {
    std::vector<std::string> result;
    for (const io::Bytes bytes : message.messages(field)) result.push_back(bytes.text(0, bytes.size));
    return result;
}

std::map<std::string, Bytes> messages(const Message& message, uint32_t field) {
    std::map<std::string, Bytes> result;
    for (const io::Bytes bytes : message.messages(field)) {
        const Message entry(bytes);
        require(result.emplace(entry.string(1), entry.bytes(2)).second, "Duplicate TensorFlow map key");
    }
    return result;
}

struct Reference {
    std::string name;
    size_t output = 0;
};

Reference reference(const std::string& value) {
    const size_t first = value.find(':');
    if (first == std::string::npos) return {value};
    Reference result{value.substr(0, first)};
    const char* const start = value.data() + value.rfind(':') + 1;
    const char* const end = value.data() + value.size();
    const std::from_chars_result parsed = std::from_chars(start, end, result.output);
    require(parsed.ec == std::errc{} && parsed.ptr == end && !result.name.empty(),
            "Invalid TensorFlow edge " + value);
    return result;
}

struct Node {
    std::string op;
    std::vector<std::string> inputs;
    std::map<std::string, Bytes> attributes;

    Message attr(const std::string& key) const {
        const auto found = attributes.find(key);
        return Message(found == attributes.end() ? Bytes{} : found->second);
    }
    std::vector<std::string> operands() const {
        std::vector<std::string> result;
        for (const std::string& input : inputs) if (input.empty() || input.front() != '^') result.push_back(input);
        return result;
    }
};

using Nodes = std::map<std::string, Node>;

Nodes nodes(const Message& graph, uint32_t field) {
    Nodes result;
    for (const io::Bytes bytes : graph.messages(field)) {
        const Message node(bytes);
        const std::string name = node.string(1);
        require(!name.empty() && result.emplace(name, Node{node.string(2), strings(node, 3), messages(node, 5)}).second,
                "Duplicate or unnamed TensorFlow node");
    }
    return result;
}

std::vector<std::string> argument_names(const Message& signature, uint32_t field) {
    std::vector<std::string> names;
    for (const io::Bytes bytes : signature.messages(field)) names.push_back(Message(bytes).string(1));
    return names;
}

struct Function {
    std::vector<std::string> inputs, outputs;
    Nodes nodes;
    std::map<std::string, std::string> results;
};

std::map<std::string, Function> functions(const Message& graph) {
    std::map<std::string, Function> result;
    for (const io::Bytes bytes : Message(graph.bytes(2)).messages(1)) {
        const Message definition(bytes), signature(definition.bytes(1));
        Function func{argument_names(signature, 2), argument_names(signature, 3), nodes(definition, 3), {}};
        for (const io::Bytes record : definition.messages(4)) {
            const Message entry(record);
            require(func.results.emplace(entry.string(1), entry.string(2)).second, "Duplicate TensorFlow function result");
        }
        require(result.emplace(signature.string(1), std::move(func)).second, "Duplicate TensorFlow function");
    }
    return result;
}

std::string function_name(const Node& node) {
    if (node.op == "PartitionedCall" || node.op == "StatefulPartitionedCall")
        return Message(node.attr("f").bytes(10)).string(1);
    return node.op;
}

std::map<std::string, std::string> restored_variables(const Nodes& nodes,
                                                       const std::map<std::string, std::string>& aliases) {
    std::map<std::string, std::string> keys;
    for (const Nodes::value_type& entry : nodes) {
        const Node& node = entry.second;
        if (node.op != "AssignVariableOp" && node.op != "Assign") continue;
        const std::vector<std::string> args = node.operands();
        if (args.size() != 2) continue;
        Reference resource = reference(args.front());
        if (const auto alias = aliases.find(resource.name); alias != aliases.end()) resource = reference(alias->second);
        Reference value = reference(args.back());
        std::set<std::string> seen;
        while (seen.insert(value.name).second) {
            const auto found = nodes.find(value.name);
            if (found == nodes.end()) break;
            const Node& producer = found->second;
            if (producer.op == "Identity") {
                value = reference(producer.operands().at(0));
                continue;
            }
            if (producer.op == "RestoreV2") {
                const std::vector<std::string> fields = producer.operands();
                const Node& constant = nodes.at(reference(fields.at(1)).name);
                const std::vector<std::string> names = strings(Message(constant.attr("value").bytes(8)), 8);
                require(constant.op == "Const" && value.output < names.size(), "Invalid TensorFlow restore tensor names");
                require(keys.emplace(resource.name, names[value.output]).second,
                        "Ambiguous TensorFlow checkpoint restore for " + resource.name);
            }
            break;
        }
    }
    return keys;
}

std::map<std::string, std::string> variable_keys(const Nodes& graph,
                        const std::map<std::string, Function>& functions, const std::string& restore) {
    std::map<std::string, std::string> keys = restored_variables(graph, {});
    for (const std::pair<const std::string, Node>& entry : graph) {
        const std::string& name = entry.first;
        const Node& node = entry.second;
        if (!restore.empty() && name != reference(restore).name) continue;
        const auto found = functions.find(function_name(node));
        if (found == functions.end()) continue;
        const Function& function = found->second;
        if (std::none_of(function.nodes.begin(), function.nodes.end(),
                         [](const Nodes::value_type& entry) { return entry.second.op == "RestoreV2"; })) continue;
        const std::vector<std::string> args = node.operands();
        require(args.size() == function.inputs.size(), "Invalid TensorFlow restore signature");
        std::map<std::string, std::string> aliases;
        for (size_t i = 0; i < args.size(); ++i) aliases.emplace(function.inputs[i], args[i]);
        keys.merge(restored_variables(function.nodes, aliases));
    }
    return keys;
}

std::vector<int> int_list(const Node& node, const char* attr) {
    std::vector<int> result;
    for (const uint64_t value : Message(node.attr(attr).bytes(1)).numbers(3)) {
        require(value <= uint64_t(INT32_MAX), "TensorFlow spatial attribute is out of range");
        result.push_back(int(value));
    }
    return result;
}

std::string data_format(const Node& node) {
    const std::string layout = node.attr("data_format").string(2);
    require(layout.empty() || layout == "NHWC" || layout == "NCHW", "Unsupported TensorFlow data format " + layout);
    return layout.empty() ? "NHWC" : layout;
}

TensorSpec spec(const Message& info) {
    const Message shape(info.bytes(3));
    return {tf::dtype(info.integer(2)), tf::dimensions(shape, true), !info.field(3) || shape.integer(3) != 0};
}

class Lowering {
public:
    Lowering(GraphBuilder& builder, std::filesystem::path prefix, std::map<std::string, Function> functions,
             std::map<std::string, std::string> keys)
        : graph_(builder), prefix_(std::move(prefix)), functions_(std::move(functions)), keys_(std::move(keys)) {}

    struct Frame {
        const Nodes& nodes;
        std::map<std::string, std::vector<std::string>> values;
        std::set<std::string> visiting, controls;
    };

    std::string value(Frame& frame, const std::string& edge, size_t depth = 0) {
        require(depth < 384, "TensorFlow graph nesting exceeds the importer limit");
        const Reference connection = reference(edge);
        auto found = frame.values.find(connection.name);
        if (found == frame.values.end()) {
            require(frame.visiting.insert(connection.name).second, "Cycle in TensorFlow graph");
            const auto node = frame.nodes.find(connection.name);
            require(node != frame.nodes.end(), "Unknown TensorFlow tensor " + connection.name);
            try {
                found = frame.values.emplace(connection.name, compile(frame, connection.name, node->second, depth)).first;
            } catch (const std::exception& error) {
                throw std::invalid_argument(connection.name + " (" + node->second.op + "): " + error.what());
            }
            frame.visiting.erase(connection.name);
        }
        require(connection.output < found->second.size(), "TensorFlow output index exceeds node outputs");
        return found->second[connection.output];
    }

private:
    void control(Frame& frame, const std::string& edge, size_t depth) {
        require(depth < 384, "TensorFlow control edges exceed the importer nesting limit");
        const std::string name = reference(edge).name;
        if (!frame.controls.insert(name).second) return;
        const Node& node = frame.nodes.at(name);
        if (node.op == "NoOp") {
            for (const std::string& input : node.inputs) {
                require(!input.empty() && input.front() == '^', "Invalid TensorFlow NoOp input");
                control(frame, input.substr(1), depth + 1);
            }
            return;
        }
        require(!frame.visiting.count(name), "Cycle in TensorFlow control edges");
        value(frame, name, depth + 1);
    }

    std::string spatial(const Node& node, std::vector<std::string> data) {
        const std::vector<int> strides = int_list(node, "strides");
        const bool channels_first = data_format(node) == "NCHW";
        const int h = channels_first ? 2 : 1, w = h + 1, channel = channels_first ? 1 : 3;
        const std::string padding = node.attr("padding").string(2);
        require(strides.size() == 4 && strides[0] == 1 && strides[channel] == 1 && strides[h] > 0 && strides[w] > 0 &&
                (padding == "SAME" || padding == "VALID"), "Unsupported TensorFlow spatial stride or padding");
        require(!data.empty(), "Missing TensorFlow spatial input");
        Json::Object attrs = Json::Object{{"stride_h", strides[h]}, {"stride_w", strides[w]}, {"same", padding == "SAME"}};
        if (channels_first) data[0] = graph_.operation("Transpose", {data[0]}, {{"perm", Json::Array{0, 2, 3, 1}}});
        const bool pool = node.op == "MaxPool" || node.op == "AvgPool";
        if (pool) {
            const std::vector<int> kernel = int_list(node, "ksize");
            require(data.size() == 1 && kernel.size() == 4 && kernel[0] == 1 && kernel[channel] == 1 &&
                    kernel[h] > 0 && kernel[h] == kernel[w], "Unsupported TensorFlow pool filter");
            attrs.emplace("maximum", node.op == "MaxPool");
            attrs.emplace("kernel", kernel[h]);
        } else {
            const std::vector<int> dilations = int_list(node, "dilations");
            require(data.size() == 2 && (dilations.empty() || dilations == std::vector<int>{1, 1, 1, 1}),
                    "TensorFlow convolution requires unit dilation");
            const bool depthwise = node.op == "DepthwiseConv2dNative";
            attrs.emplace("depthwise", depthwise);
            data[1] = depthwise ? graph_.operation("DepthwiseFilter", {data[1]}) :
                      graph_.operation("Transpose", {data[1]}, {{"perm", Json::Array{3, 0, 1, 2}}});
        }
        std::string result = graph_.operation(pool ? "NhwcPool" : "NhwcConv", data, std::move(attrs));
        return channels_first ? graph_.operation("Transpose", {result}, {{"perm", Json::Array{0, 3, 1, 2}}}) : result;
    }

    int constant_index(const std::string& value) const {
        const Tensor* item = graph_.constant_value(value);
        require(item && item->numel() == 1 && (item->dtype() == DType::Int32 || item->dtype() == DType::Int64),
                "TensorFlow axis requires a constant integer");
        const int64_t index = item->dtype() == DType::Int32 ? item->int_data()[0] : item->int64_data()[0];
        require(index >= INT32_MIN && index <= INT32_MAX, "TensorFlow axis cannot be represented");
        return int(index);
    }

    std::string variable(const std::string& name, const Node& node) {
        if (!bundle_) bundle_.emplace(prefix_);
        const auto key = keys_.find(name);
        const std::string shared = node.attr("shared_name").string(2);
        std::string selected = key != keys_.end() ? key->second : (shared.empty() ? name : shared);
        if (key == keys_.end() && !bundle_->contains(selected) && bundle_->contains(selected + "/.ATTRIBUTES/VARIABLE_VALUE"))
            selected += "/.ATTRIBUTES/VARIABLE_VALUE";
        return graph_.constant(bundle_->tensor(selected, graph_.backend()));
    }

    std::vector<std::string> compile(Frame& frame, const std::string& name, const Node& node, size_t depth) {
        const std::string& op = node.op;
        if (op == "Const") return {graph_.constant(tf::constant(Message(node.attr("value").bytes(8)), graph_.backend()))};
        if (op == "VarHandleOp" || op == "VariableV2") return {variable(name, node)};
        if (op == "Placeholder") throw std::invalid_argument("TensorFlow placeholder is absent from the selected signature");
        std::vector<std::string> data;
        for (const std::string& edge : node.operands()) data.push_back(value(frame, edge, depth + 1));
        for (const std::string& edge : node.inputs)
            if (!edge.empty() && edge.front() == '^') control(frame, edge.substr(1), depth + 1);
        const auto function = functions_.find(function_name(node));
        if (function != functions_.end()) {
            const Function& definition = function->second;
            require(data.size() == definition.inputs.size(), "TensorFlow function input count differs");
            Frame body{definition.nodes, {}, {}, {}};
            for (size_t i = 0; i < data.size(); ++i) body.values.emplace(definition.inputs[i], std::vector<std::string>{data[i]});
            std::vector<std::string> results;
            for (const std::string& key : definition.outputs) results.push_back(value(body, definition.results.at(key), depth + 1));
            return results;
        }
        if (op == "Identity" || op == "ReadVariableOp" || op == "StopGradient" || op == "Snapshot")
            return {data.at(0)};
        if (op == "IdentityN") return data;
        if (op == "Conv2D" || op == "DepthwiseConv2dNative" || op == "MaxPool" || op == "AvgPool")
            return {spatial(node, std::move(data))};
        if (op == "BiasAdd") {
            if (data_format(node) == "NCHW")
                return {graph_.operation("ChannelBias", data)};
            return {graph_.operation("Add", data)};
        }
        if (op == "FusedBatchNorm" || op == "FusedBatchNormV2" || op == "FusedBatchNormV3") {
            require(data.size() == 5 && !node.attr("is_training").integer(5, 1), "TensorFlow training batch normalization is unsupported");
            const uint32_t bits = uint32_t(node.attr("epsilon").integer(4, 0x38d1b717));
            float epsilon;
            std::memcpy(&epsilon, &bits, sizeof(epsilon));
            return {graph_.operation("AffineBatchNormalization", {data[0], data[3], data[4], data[1], data[2]},
                                     {{"axis", data_format(node) == "NHWC" ? -1 : 1}, {"epsilon", double(epsilon)}})};
        }
        if (op == "MatMul" || op == "BatchMatMul" || op == "BatchMatMulV2") {
            const bool routine = op == "MatMul";
            return {graph_.operation("MatMul", data, {{"transA", node.attr(routine ? "transpose_a" : "adj_x").integer(5) != 0},
                                                     {"transB", node.attr(routine ? "transpose_b" : "adj_y").integer(5) != 0},
                                                     {"adjoint", !routine}})};
        }
        if (op == "Cast") return {graph_.operation("Cast", data, {{"to", dtype_name(tf::dtype(node.attr("DstT").integer(6)))}})};
        if (op == "Reshape") return {graph_.operation(op, data, {{"allowzero", 1}})};
        if (op == "Mean" || op == "Sum")
            return {graph_.operation(op, data, {{"keep_dims", node.attr("keep_dims").integer(5) != 0}})};
        if (op == "Shape")
            return {graph_.operation(op, data, {{"to", dtype_name(tf::dtype(node.attr("out_type").integer(6, 3)))}})};
        if (op == "ConcatV2" || op == "Concat") {
            require(data.size() >= 2, "Invalid TensorFlow concat input count");
            const auto axis_position = op == "ConcatV2" ? data.end() - 1 : data.begin();
            const int axis = constant_index(*axis_position);
            data.erase(axis_position);
            return {graph_.operation("Concat", data, {{"axis", axis}})};
        }
        if (op == "GatherV2") {
            require(data.size() == 3 && !node.attr("batch_dims").integer(3), "Unsupported TensorFlow batched gather");
            return {graph_.operation("Gather", {data[0], data[1]}, {{"axis", constant_index(data[2])}})};
        }
        if (op == "Squeeze") {
            const std::vector<uint64_t> axes = Message(node.attr("squeeze_dims").bytes(1)).numbers(3);
            Json::Array values;
            for (uint64_t dim : axes) values.emplace_back(int(int64_t(dim)));
            return {graph_.operation(op, data, {{"axes", std::move(values)}})};
        }
        if (op == "Relu6") return {graph_.operation("Clip", data, {{"min", 0}, {"max", 6}})};
        if (op == "LeakyRelu") {
            const uint32_t bits = uint32_t(node.attr("alpha").integer(4, 0x3e4ccccd));
            float alpha;
            std::memcpy(&alpha, &bits, sizeof(alpha));
            return {graph_.operation(op, data, {{"alpha", double(alpha)}})};
        }
        static const std::map<std::string, std::string> operations{
            {"Add", "Add"}, {"AddV2", "Add"}, {"Sub", "Sub"}, {"Mul", "Mul"}, {"Div", "Div"}, {"RealDiv", "Div"},
            {"Relu", "Relu"}, {"Sigmoid", "Sigmoid"}, {"Tanh", "Tanh"}, {"Softmax", "Softmax"},
            {"LogSoftmax", "LogSoftmax"}, {"Abs", "Abs"}, {"Neg", "Neg"}, {"Square", "Square"},
            {"Minimum", "Minimum"}, {"Maximum", "Maximum"}, {"Exp", "Exp"}, {"Log", "Log"},
            {"Sqrt", "Sqrt"}, {"Rsqrt", "Rsqrt"}, {"Transpose", "Transpose"}, {"Gather", "Gather"},
            {"ExpandDims", "ExpandDims"}, {"Pack", "Stack"}, {"StridedSlice", "StridedSlice"}, {"Pad", "Pad"},
        };
        const auto found = operations.find(op);
        require(found != operations.end(), "Unsupported TensorFlow inference operator " + op);
        Json::Object attributes;
        if (op == "Pack") attributes.emplace("axis", int(node.attr("axis").integer(3)));
        if (op == "StridedSlice") {
            for (const char* field : {"begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask"})
                attributes.emplace(field, int(node.attr(field).integer(3)));
        }
        return {graph_.operation(found->second, data, std::move(attributes))};
    }

    GraphBuilder& graph_;
    std::filesystem::path prefix_;
    std::map<std::string, Function> functions_;
    std::map<std::string, std::string> keys_;
    std::optional<tf::Bundle> bundle_;
};

}  // namespace

InferenceSession load_saved_model_session(const std::filesystem::path& path, Backend backend, int device,
                                          const std::string& signature_name) {
    DeviceScope owner(backend, device);
    const std::filesystem::path file = std::filesystem::is_directory(path) ? path / "saved_model.pb" : path;
    io::BinaryInput source(file);
    const std::vector<uint8_t> contents = source.bytes(source.size());
    const Message saved(io::view(contents));
    require(saved.integer(1) == 1, "Unsupported TensorFlow SavedModel schema version");
    Bytes model_data, signature_data;
    bool serving = false;
    for (const io::Bytes data : saved.messages(2)) {
        const Message candidate(data);
        const std::map<std::string, Bytes> signatures = messages(candidate, 5);
        const auto match = signatures.find(signature_name);
        if (match == signatures.end()) continue;
        const std::vector<std::string> tags = strings(Message(candidate.bytes(1)), 4);
        const bool serve = std::find(tags.begin(), tags.end(), "serve") != tags.end();
        if (!model_data.size || (serve && !serving)) {
            model_data = data;
            signature_data = match->second;
            serving = serve;
        } else if (serve == serving) {
            throw std::invalid_argument("Ambiguous SavedModel signature " + signature_name);
        }
    }
    require(model_data.size != 0, "SavedModel signature not found: " + signature_name);
    const Message model(model_data), graph(model.bytes(2)), signature(signature_data);
    const Nodes network = nodes(graph, 1);
    std::map<std::string, Function> library = functions(graph);
    std::map<std::string, std::string> keys = variable_keys(network, library, Message(model.bytes(3)).string(3));
    GraphBuilder builder(backend, device);
    Lowering lowering(builder, file.parent_path() / "variables" / "variables", std::move(library), std::move(keys));
    Lowering::Frame frame{network, {}, {}, {}};
    for (const std::pair<const std::string, io::Bytes>& entry : messages(signature, 1)) {
        const std::string& name = entry.first;
        const io::Bytes& info = entry.second;
        const Message port(info);
        const Reference edge = reference(port.string(1));
        require(!edge.name.empty() && !edge.output &&
                frame.values.emplace(edge.name, std::vector<std::string>{builder.input(name, spec(port))}).second,
                "Unsupported TensorFlow input signature edge");
    }
    for (const std::pair<const std::string, io::Bytes>& entry : messages(signature, 2)) {
        const std::string& name = entry.first;
        const io::Bytes& info = entry.second;
        const Message port(info);
        builder.output(name, lowering.value(frame, port.string(1)), spec(port));
    }
    return std::move(builder).finish();
}

}  // namespace elman
