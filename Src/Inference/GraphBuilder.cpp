#include "Inference/GraphBuilder.h"

#include <stdexcept>

namespace elman {
namespace {

Json::Object json_spec(const TensorSpec& spec) {
  Json::Object values;
  if (!spec.any_dtype) values.emplace("dtype", dtype_name(spec.dtype));
  if (!spec.any_rank) values.emplace("shape", Json::Array(spec.shape.begin(), spec.shape.end()));
  return values;
}

}  // namespace

std::string GraphBuilder::next_value() { return "_elman_graph/" + std::to_string(counter_++); }

std::string GraphBuilder::input(const std::string& name, const TensorSpec& spec) {
  const std::string value = next_value();
  Json::Object description = json_spec(spec);
  description.emplace("value", value);
  if (!inputs_.emplace(name, std::move(description)).second)
    throw std::invalid_argument("Duplicate graph input port " + name);
  return value;
}

std::string GraphBuilder::constant(Tensor tensor) {
  const std::string name = next_value();
  weights_.emplace(name, std::move(tensor));
  return name;
}

std::string GraphBuilder::operation(std::string op, const std::vector<std::string>& inputs, Json::Object attributes) {
  const std::string value = next_value();
  const std::string name = op + " " + value;
  nodes_.emplace_back(Json::Object{{"name", name}, {"op", std::move(op)}, {"output", value},
     {"inputs", Json::Array(inputs.begin(), inputs.end())}, {"attributes", std::move(attributes)}});
  return value;
}

void GraphBuilder::output(const std::string& name, const std::string& value, const TensorSpec& spec) {
  Json::Object description = json_spec(spec);
  description.emplace("value", value);
  if (!outputs_.emplace(name, std::move(description)).second)
    throw std::invalid_argument("Duplicate graph output port " + name);
}

InferenceSession GraphBuilder::finish() && {
  return make_graph_session(Json::Object{{"inputs", std::move(inputs_)}, {"outputs", std::move(outputs_)},
                                         {"nodes", std::move(nodes_)}}, std::move(weights_), backend_, device_);
}

const Tensor* GraphBuilder::constant_value(const std::string& name) const {
  const auto found = weights_.find(name);
  return found == weights_.end() ? nullptr : &found->second;
}

}  // namespace elman
