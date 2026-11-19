#include "NN/Module.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Core/Graph.h"

namespace elman::nn {

void Module::validate_name(const std::string& name) const {
  if (name.empty() || name.find('.') != std::string::npos ||
      std::any_of(parameters_.begin(), parameters_.end(), [&](const auto& item) { return item.name == name; }) ||
      std::any_of(buffers_.begin(), buffers_.end(), [&](const auto& item) { return item.tensor.name == name; }) ||
      std::any_of(children_.begin(), children_.end(), [&](const auto& item) { return item.name == name; }))
    throw std::invalid_argument("Invalid or duplicate module name: " + name);
}

Tensor Module::register_parameter(std::string name, Tensor value) {
  validate_name(name);
  if (!value.defined() || !value.node()->parents.empty())
    throw std::invalid_argument("Module parameter must be a leaf tensor: " + name);
  parameters_.push_back({std::move(name), value});
  return value;
}

Tensor Module::register_buffer(std::string name, Tensor value, bool persistent) {
  validate_name(name);
  if (!value.defined() || value.requires_grad())
    throw std::invalid_argument("Module buffer must be a defined non-gradient tensor: " + name);
  buffers_.push_back({{std::move(name), value}, persistent});
  return value;
}

Module& Module::register_module(std::string name, std::unique_ptr<Module> child) {
  validate_name(name);
  if (!child) throw std::invalid_argument("Missing child module: " + name);
  child->train(training_);
  children_.push_back({std::move(name), std::move(child)});
  return *children_.back().module;
}

void Module::train(bool enabled) {
  training_ = enabled;
  for (auto& child : children_) child.module->train(enabled);
}

void Module::collect(std::vector<NamedTensor>& result, const std::string& prefix,
                     bool parameters, bool persistent_only) const {
  if (parameters) {
    for (const auto& item : parameters_) result.push_back({prefix + item.name, item.value});
  } else {
    for (const auto& item : buffers_) {
      if (!persistent_only || item.persistent) result.push_back({prefix + item.tensor.name, item.tensor.value});
    }
  }
  for (const auto& child : children_) child.module->collect(result, prefix + child.name + '.', parameters, persistent_only);
}

std::vector<NamedTensor> Module::named_parameters() const {
  std::vector<NamedTensor> result;
  collect(result, {}, true, false);
  std::unordered_set<const TensorNode*> seen;
  result.erase(std::remove_if(result.begin(), result.end(), [&](const auto& item) {
    return !seen.insert(item.value.node().get()).second;
  }), result.end());
  return result;
}

std::vector<NamedTensor> Module::named_buffers(bool persistent_only) const {
  std::vector<NamedTensor> result;
  collect(result, {}, false, persistent_only);
  return result;
}

StateDict Module::state_dict() const {
  std::vector<NamedTensor> items;
  collect(items, {}, true, false);
  collect(items, {}, false, true);
  StateDict state;
  for (const auto& item : items) {
    const auto& tensor = item.value;
    state.emplace(item.name, Tensor::from_raw(tensor.shape(), tensor.dtype(), tensor.raw_data()));
  }
  return state;
}

void Module::load_state_dict(const StateDict& state) {
  std::vector<NamedTensor> items;
  collect(items, {}, true, false);
  collect(items, {}, false, true);
  if (state.size() != items.size()) throw std::invalid_argument("Unexpected module state entry count");
  struct Update { Tensor tensor; std::vector<uint8_t> bytes; };
  std::vector<Update> updates;
  std::unordered_map<const TensorNode*, size_t> aliases;
  for (const auto& item : items) {
    const auto supplied = state.find(item.name);
    const auto& target = item.value;
    if (supplied == state.end() || !supplied->second.defined() ||
        supplied->second.shape() != target.shape() || supplied->second.dtype() != target.dtype())
      throw std::invalid_argument("Missing or incompatible module state entry: " + item.name);
    auto bytes = supplied->second.raw_data();
    auto [position, inserted] = aliases.emplace(target.node().get(), updates.size());
    if (inserted) updates.push_back({target, std::move(bytes)});
    else if (updates[position->second].bytes != bytes)
      throw std::invalid_argument("Conflicting state for tied module tensor: " + item.name);
  }
  for (auto& update : updates) update.tensor.assign_raw(update.bytes);
}

}  // namespace elman::nn
