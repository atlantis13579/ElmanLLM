#include "Models/GraphModel.h"

#include <set>
#include <stdexcept>

#include "Core/Autocast.h"
#include "Core/Graph.h"

namespace elman {
namespace {
void validate_tensor(const Json& spec, const Tensor& value, Backend backend) {
  if (!value.defined() || value.backend() != backend ||
      value.dtype() != parse_dtype(spec.at("dtype").string())) {
    throw std::invalid_argument("Graph port dtype/backend mismatch: " +
                                spec.at("name").string());
  }
  const auto& dims = spec.at("shape").array();
  if (dims.size() != value.shape().size()) {
    throw std::invalid_argument("Graph port rank mismatch");
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    const int dim = dims[i].integer();
    if (value.shape()[i] <= 0 || (dim != -1 && dim != value.shape()[i])) {
      throw std::invalid_argument("Graph port shape mismatch");
    }
    if (spec.contains("max_shape") &&
        spec.at("max_shape").at(i).integer() > 0 &&
        value.shape()[i] > spec.at("max_shape").at(i).integer()) {
      throw std::invalid_argument("Graph port exceeds maximum shape");
    }
  }
}
}  // namespace

GraphModel::GraphModel(const Json& definition, Backend backend, uint32_t seed)
    : Model(definition.at("name").string(),
            definition.string_or("architecture", "graph"), backend, seed),
      definition_(nn::resolve_model_config(definition)) {
  task_ = definition_.at("task").string();
  if (task_ != "regression" && task_ != "classification" &&
      task_ != "language_modeling") {
    throw std::invalid_argument("Unknown graph task");
  }
  classes_ = definition_.at("output_features").integer();
  if (classes_ < (regression() ? 1 : 2)) {
    throw std::invalid_argument("Invalid graph output features");
  }
  Json::Array inputs, outputs;
  for (const auto& key : {"inputs", "outputs"}) {
    auto& ports = key == std::string("inputs") ? inputs : outputs;
    std::set<std::string> names;
    for (const auto& port : definition_.at(key).array()) {
      const auto name = port.at("name").string();
      if (name.empty() || name[0] == '@' || !names.insert(name).second) {
        throw std::invalid_argument("Invalid/duplicate graph port name");
      }
      parse_dtype(port.at("dtype").string());
      for (const auto& dim : port.at("shape").array()) {
        if (dim.integer() == 0 || dim.integer() < -1) {
          throw std::invalid_argument("Invalid graph port dimension");
        }
      }
      if (port.contains("max_shape") && port.at("max_shape").array().size() !=
                                            port.at("shape").array().size()) {
        throw std::invalid_argument("Maximum shape rank mismatch");
      }
      ports.push_back(port.at(key == std::string("inputs") ? "name" : "value"));
    }
    if (ports.empty()) {
      throw std::invalid_argument("Graph requires nonempty input/output ports");
    }
  }
  const auto& input = definition_.at("inputs").at(0);
  const auto& sample = input.contains("sample_shape") ? input.at("sample_shape")
                                                      : input.at("shape");
  for (size_t i = input.contains("sample_shape") ? 0 : 1;
       i < sample.array().size(); ++i) {
    if (sample.at(i).integer() <= 0) {
      throw std::invalid_argument(
          "Dynamic input requires positive sample_shape");
    }
    sample_shape_.push_back(sample.at(i).integer());
  }
  graph_ = std::make_unique<nn::GraphModule>(
      Json::Object{{"inputs", inputs},
                   {"outputs", outputs},
                   {"layers", definition_.at("layers")}},
      backend, seed);
  for (const auto& parameter : graph_->named_parameters()) {
    parameters_.push_back({parameter.name, parameter.value});
  }
}

nn::TensorMap GraphModel::forward(const nn::TensorMap& inputs, bool training,
                                  uint32_t seed) {
  if (inputs.size() != definition_.at("inputs").array().size()) {
    throw std::invalid_argument("Graph input count mismatch");
  }
  for (const auto& port : definition_.at("inputs").array()) {
    validate_tensor(port, inputs.at(port.at("name").string()), backend_);
  }
  graph_->train(training);
  std::unique_ptr<graph::RecordingGuard> guard;
  if (!training) {
    guard = std::make_unique<graph::RecordingGuard>();
  }
  const auto values = graph_->run(inputs, seed);
  nn::TensorMap result;
  for (const auto& port : definition_.at("outputs").array()) {
    auto value = values.at(port.at("value").string());
    if (training && autocast::dtype() != DType::Float32 &&
        value.dtype() == autocast::dtype() &&
        parse_dtype(port.at("dtype").string()) == DType::Float32) {
      value = cast(value, DType::Float32);
    }
    validate_tensor(port, value, backend_);
    if (!result
             .emplace(port.at("name").string(),
                      training ? value : value.detach())
             .second) {
      throw std::invalid_argument("Duplicate output port");
    }
  }
  return result;
}

Tensor GraphModel::forward(const Tensor& input, bool training, uint32_t seed) {
  if (definition_.at("inputs").array().size() != 1 ||
      definition_.at("outputs").array().size() != 1) {
    throw std::invalid_argument("Use named inputs for a multi-port graph");
  }
  return forward(
             nn::TensorMap{
                 {definition_.at("inputs").at(0).at("name").string(), input}},
             training, seed)
      .begin()
      ->second;
}

std::vector<Parameter> GraphModel::checkpoint_buffers() const {
  std::vector<Parameter> result;
  for (const auto& buffer : graph_->named_buffers(true)) {
    result.push_back({buffer.name, buffer.value});
  }
  return result;
}
}  // namespace elman
