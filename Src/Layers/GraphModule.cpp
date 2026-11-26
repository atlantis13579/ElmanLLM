#include "Layers/GraphModule.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include "Core/TensorDispatch.h"
#include "Layers/Layers.h"
#include "Fusion/Fusion.h"

namespace elman::nn {
namespace {
std::vector<std::string> strings(const Json& values) {
  std::vector<std::string> result;
  for (const auto& value : values.array()) {
    result.push_back(value.string());
  }
  return result;
}
Shape shape(const Json& values) {
  Shape result;
  for (const auto& value : values.array()) {
    result.push_back(value.integer());
  }
  return result;
}

class ParameterLayer final : public Module {
 public:
  ParameterLayer(const Json& config, Backend backend, uint32_t& seed) {
    const Shape dims = shape(config.at("shape"));
    shape_element_count(dims);
    const auto kind = config.string_or("initializer", "normal");
    Tensor value;
    if (kind == "normal") {
      value = Tensor::normal(dims, config.number_or("mean", 0),
                             config.number_or("stddev", 0.02), seed++, backend,
                             true);
    } else if (kind == "zeros" || kind == "ones") {
      value = kind == "zeros" ? Tensor::zeros(dims, backend, true)
                              : Tensor::ones(dims, backend, true);
    } else {
      throw std::invalid_argument("Unknown parameter initializer: " + kind);
    }
    weight_ = register_parameter("weight", value);
  }

 private:
  Tensor forward(const Tensor&, uint32_t) override {
    return weight_;
  }
  Tensor weight_;
};

class LayerNormalization final : public Module {
 public:
  LayerNormalization(const Json& config, Backend backend)
      : epsilon_(config.number_or("epsilon", 1e-5)) {
    const int width = config.at("features").integer();
    if (width <= 0 || epsilon_ <= 0) {
      throw std::invalid_argument("Invalid LayerNorm dimensions/epsilon");
    }
    weight_ =
        register_parameter("weight", Tensor::ones({width}, backend, true));
    bias_ = register_parameter("bias", Tensor::zeros({width}, backend, true));
  }

 private:
  Tensor forward(const Tensor& input, uint32_t) override {
    return layer_norm(input, weight_, bias_, epsilon_);
  }
  Tensor weight_, bias_;
  double epsilon_;
};

Json resolve(const Json& value, const Json& parameters, int depth = 0) {
  if (depth > 64) {
    throw std::invalid_argument("Cyclic model parameter reference");
  }
  if (value.is_string() && !value.string().empty() &&
      value.string()[0] == '$') {
    return resolve(parameters.at(value.string().substr(1)), parameters,
                   depth + 1);
  }
  if (value.is_array()) {
    Json::Array result;
    for (const auto& item : value.array()) {
      result.push_back(resolve(item, parameters, depth + 1));
    }
    return result;
  }
  if (value.is_object()) {
    if (value.contains("sqrt")) {
      const auto result =
          std::sqrt(resolve(value.at("sqrt"), parameters, depth + 1).number());
      if (!std::isfinite(result)) {
        throw std::invalid_argument("Invalid square root expression");
      }
      return result;
    }
    if (value.contains("product") || value.contains("quotient")) {
      const bool product = value.contains("product");
      const auto args = resolve(value.at(product ? "product" : "quotient"),
                                parameters, depth + 1)
                            .array();
      if (args.empty() || (!product && args.size() != 2)) {
        throw std::invalid_argument("Invalid dimension expression");
      }
      double result = product ? 1 : args[0].number();
      if (product) {
        for (const auto& arg : args) {
          result *= arg.number();
        }
      } else {
        result /= args[1].number();
      }
      if (!std::isfinite(result)) {
        throw std::invalid_argument("Invalid dimension expression result");
      }
      return result;
    }
    Json::Object result;
    for (const auto& item : value.object()) {
      result[item.first] = resolve(item.second, parameters, depth + 1);
    }
    return result;
  }
  return value;
}
Json expand_subgraphs(const Json& value, const Json& parameters,
                      const Json& definitions, std::vector<std::string>& stack,
                      int depth = 0) {
  if (depth > 128) {
    throw std::invalid_argument("Subgraph nesting limit exceeded");
  }
  if (value.is_array()) {
    Json::Array result;
    for (const auto& item : value.array()) {
      result.push_back(
          expand_subgraphs(item, parameters, definitions, stack, depth + 1));
    }
    return result;
  }
  if (!value.is_object()) {
    return resolve(value, parameters);
  }
  if (value.contains("$include")) {
    throw std::invalid_argument(
        "Load JSON includes with read_model_definition(path)");
  }
  if (value.contains("class_name") &&
      value.at("class_name").string() == "Subgraph") {
    const auto& config = value.at("config");
    for (const auto& field : config.object()) {
      if (field.first != "ref" && field.first != "parameters") {
        throw std::invalid_argument("Unknown Subgraph config: " + field.first);
      }
    }
    const auto ref = config.at("ref").string();
    if (!definitions.contains(ref)) {
      throw std::invalid_argument("Unknown subgraph definition: " + ref);
    }
    if (std::find(stack.begin(), stack.end(), ref) != stack.end()) {
      throw std::invalid_argument("Recursive subgraph definition: " + ref);
    }
    const auto& body = definitions.at(ref);
    if (body.contains("$include")) {
      throw std::invalid_argument(
          "Load JSON includes with read_model_definition(path)");
    }
    const auto type = body.at("class_name").string();
    if (type != "Graph" && type != "Sequence" && type != "Sequential") {
      throw std::invalid_argument(
          "Subgraph definition must be Graph or Sequence: " + ref);
    }
    auto local = body.contains("parameters") ? body.at("parameters").object()
                                             : Json::Object{};
    if (config.contains("parameters")) {
      for (const auto& argument : config.at("parameters").object()) {
        if (!local.count(argument.first)) {
          throw std::invalid_argument("Unknown subgraph parameter: " + ref +
                                      "." + argument.first);
        }
        local[argument.first] = resolve(argument.second, parameters);
      }
    }
    for (const auto& argument : local) {
      if (argument.second.is_null()) {
        throw std::invalid_argument("Missing subgraph parameter: " + ref + "." +
                                    argument.first);
      }
    }
    stack.push_back(ref);
    auto result = value.object();
    result["class_name"] = type;
    result["config"] = expand_subgraphs(body.at("config"), local, definitions,
                                        stack, depth + 1);
    stack.pop_back();
    return result;
  }
  if (value.contains("sqrt") || value.contains("product") ||
      value.contains("quotient")) {
    return resolve(value, parameters);
  }
  Json::Object result;
  for (const auto& item : value.object()) {
    // Definitions remain templates until instantiated in their own parameter
    // scope.
    if (item.first != "definitions") {
      result[item.first] = expand_subgraphs(item.second, parameters,
                                            definitions, stack, depth + 1);
    }
  }
  return result;
}
}  // namespace

Json resolve_model_config(const Json& definition) {
  std::vector<std::string> stack;
  return expand_subgraphs(
      definition,
      definition.contains("parameters") ? definition.at("parameters")
                                        : Json(Json::Object{}),
      definition.contains("definitions") ? definition.at("definitions")
                                         : Json(Json::Object{}),
      stack);
}

std::unique_ptr<Module> create_layer(const std::string& type, const Json& c,
                                     Backend backend, uint32_t& seed) {
  if (type == "Graph" || type == "Sequence" || type == "Sequential") {
    auto config = c.object();
    config["class_name"] = type;
    return std::make_unique<GraphModule>(config, backend, seed);
  }
  if (type == "Linear" || type == "Conv2d") {
    const uint32_t layer_seed = seed;
    seed += 2;
    if (type == "Linear") {
      return std::make_unique<Linear>(c.at("in_features").integer(),
                                      c.at("out_features").integer(), backend,
                                      layer_seed, c.bool_or("bias", true));
    }
    return std::make_unique<Conv2d>(
        c.at("in_channels").integer(), c.at("out_channels").integer(),
        c.at("kernel_size").integer(), backend, layer_seed,
        c.int_or("stride", 1), c.int_or("padding", 0), c.bool_or("bias", true));
  }
  if (type == "GroupNorm") {
    return std::make_unique<GroupNorm>(c.at("groups").integer(),
                                       c.at("channels").integer(), backend,
                                       c.number_or("epsilon", 1e-5));
  }
  if (type == "BatchNorm2d") {
    return std::make_unique<BatchNorm2d>(c.at("channels").integer(), backend,
                                         c.number_or("epsilon", 1e-5),
                                         c.number_or("momentum", 0.1));
  }
  if (type == "LayerNorm") {
    return std::make_unique<LayerNormalization>(c, backend);
  }
  if (type == "Parameter") {
    return std::make_unique<ParameterLayer>(c, backend, seed);
  }
  if (type == "ReLU") {
    return std::make_unique<ReLU>();
  }
  if (type == "Dropout") {
    return std::make_unique<Dropout>(c.number_or("probability", 0.5));
  }
  return nullptr;
}

GraphModule::GraphModule(const Json& definition, Backend backend,
                         uint32_t& seed)
    : backend_(backend) {
  const bool sequence = definition.string_or("class_name", "Graph") != "Graph";
  inputs_ = sequence ? std::vector<std::string>{"input"}
                     : strings(definition.at("inputs"));
  if (inputs_.empty()) {
    throw std::invalid_argument("Graph requires inputs");
  }
  std::set<std::string> values, names;
  for (const auto& input : inputs_) {
    if (input.empty() || !values.insert(input).second) {
      throw std::invalid_argument("Duplicate/empty graph input");
    }
  }
  std::vector<Node> pending;
  std::string previous = inputs_[0];
  const int repeats = sequence ? definition.int_or("repeat", 1) : 1;
  if (repeats <= 0 || repeats > 10000) {
    throw std::invalid_argument("Invalid Sequence repeat count");
  }
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (const auto& layer : definition.at("layers").array()) {
      Node node;
      node.name = layer.at("name").string();
      if (repeats > 1) {
        node.name += "_" + std::to_string(repeat);
      }
      node.type = layer.at("class_name").string();
      if (node.name.empty() || node.name.find('.') != std::string::npos ||
          !names.insert(node.name).second) {
        throw std::invalid_argument("Duplicate/invalid graph layer: " +
                                    node.name);
      }
      node.config =
          layer.contains("config") ? layer.at("config") : Json(Json::Object{});
      node.inputs = layer.contains("inputs") ? strings(layer.at("inputs"))
                    : sequence ? std::vector<std::string>{previous}
                               : std::vector<std::string>{};
      node.outputs = layer.contains("outputs")
                         ? strings(layer.at("outputs"))
                         : std::vector<std::string>{node.name};
      for (const auto& output : node.outputs) {
        if (output.empty() || !values.insert(output).second) {
          throw std::invalid_argument("Duplicate/empty graph value: " + output);
        }
      }
      validate_operation(node.type, node.config, node.inputs.size(),
                         node.outputs.size());
      if (node.type == "Graph" &&
          (node.inputs.size() != node.config.at("inputs").array().size() ||
           node.outputs.size() != node.config.at("outputs").array().size())) {
        throw std::invalid_argument("Nested graph port count mismatch: " +
                                    node.name);
      }
      if (auto module = create_layer(node.type, node.config, backend, seed)) {
        node.module = &register_module(node.name, std::move(module));
      }
      previous = node.outputs.at(0);
      pending.push_back(std::move(node));
    }
  }
  outputs_ = sequence ? std::vector<std::string>{previous}
                      : strings(definition.at("outputs"));
  if (outputs_.empty()) {
    throw std::invalid_argument("Graph requires outputs");
  }
  // Parameters are explicit graph values, allowing shared weights without
  // copies.
  std::set<std::string> available(inputs_.begin(), inputs_.end());
  for (const auto& parameter : named_parameters()) {
    available.insert("@" + parameter.name);
  }
  while (!pending.empty()) {
    const auto next =
        std::find_if(pending.begin(), pending.end(), [&](const Node& node) {
          return std::all_of(
              node.inputs.begin(), node.inputs.end(),
              [&](const auto& input) { return available.count(input) != 0; });
        });
    if (next == pending.end()) {
      throw std::invalid_argument("Graph has a cycle or unresolved input at " +
                                  pending.front().name);
    }
    available.insert(next->outputs.begin(), next->outputs.end());
    nodes_.push_back(std::move(*next));
    pending.erase(next);
  }
  for (const auto& output : outputs_) {
    if (!available.count(output)) {
      throw std::invalid_argument("Unknown graph output: " + output);
    }
  }
  for (const auto& node : nodes_) {
    for (const auto& input : node.inputs) {
      ++readers_[input];
    }
  }
  for (const auto& output : outputs_) {
    ++readers_[output];
  }
}

TensorMap GraphModule::run(const TensorMap& inputs, uint32_t seed) {
  fusion::Scope fusion_scope;
  if (inputs.size() != inputs_.size()) {
    throw std::invalid_argument("Graph input count mismatch");
  }
  TensorMap values;
  auto readers = readers_;
  for (const auto& name : inputs_) {
    values.emplace(name, inputs.at(name));
  }
  for (const auto& parameter : named_parameters()) {
    values.emplace("@" + parameter.name, parameter.value);
  }
  for (auto& node : nodes_) {
    std::vector<Tensor> operands;
    for (const auto& name : node.inputs) {
      operands.push_back(values.at(name));
    }
    std::vector<Tensor> result;
    try {
      if (auto graph = dynamic_cast<GraphModule*>(node.module)) {
        if (operands.size() != graph->inputs_.size()) {
          throw std::invalid_argument("Nested graph input count mismatch");
        }
        TensorMap args;
        for (size_t i = 0; i < operands.size(); ++i) {
          args.emplace(graph->inputs_[i], operands[i]);
        }
        const auto outputs = graph->run(args, seed++);
        for (const auto& name : graph->outputs_) {
          result.push_back(outputs.at(name));
        }
      } else if (node.module) {
        result.push_back(
            (*node.module)(operands.empty() ? Tensor{} : operands[0], seed++));
      } else {
        result = execute_operation(node.type, operands, node.config, backend_,
                                   seed++, training());
      }
    } catch (const std::exception& error) {
      throw std::runtime_error("Layer " + node.name + " (" + node.type +
                               "): " + error.what());
    }
    if (result.size() != node.outputs.size()) {
      throw std::invalid_argument("Graph output count mismatch: " + node.name);
    }
    for (size_t i = 0; i < result.size(); ++i) {
      values.emplace(node.outputs[i], result[i]);
    }
    for (const auto& input : node.inputs) {
      if (--readers.at(input) == 0) {
        values.erase(input);
      }
    }
    for (const auto& output : node.outputs) {
      if (!readers[output]) {
        values.erase(output);
      }
    }
  }
  TensorMap result;
  for (const auto& name : outputs_) {
    result.emplace(name, values.at(name));
  }
  fusion_scope.finish();
  return result;
}

Tensor GraphModule::forward(const Tensor& input, uint32_t seed) {
  if (inputs_.size() != 1 || outputs_.size() != 1) {
    throw std::invalid_argument("Use run() for a multi-port graph");
  }
  return run({{inputs_[0], input}}, seed).at(outputs_[0]);
}
}  // namespace elman::nn
