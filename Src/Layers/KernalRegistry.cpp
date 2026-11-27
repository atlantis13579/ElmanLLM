#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "Core/Device.h"
#include "Core/TensorDispatch.h"
#include "Layers/GraphModule.h"
#include "Kernals/audio_ops.h"
#include "Kernals/flux2_ops.h"
#include "Kernals/inference_ops.h"
#include "Kernals/recurrent_ops.h"

namespace elman::nn {
namespace {
Shape integers(const Json& value) {
  Shape result;
  for (const auto& item : value.array()) {
    result.push_back(item.integer());
  }
  return result;
}
}  // namespace

const Json& kernal_catalog() {
  static const Json catalog = [] {
    // Separate literals avoid MSVC's per-literal and concatenated-string limits.
    const char* const chunks[] = {
#include "KernalCatalog.inc"
    };
    std::string text;
    for (const char* chunk : chunks) text += chunk;
    return Json::parse(text);
  }();
  return catalog;
}

void validate_operation(const std::string& type, const Json& config,
                        size_t inputs, size_t outputs) {
  const auto& entries = kernal_catalog().at("entries");
  if (!entries.contains(type)) {
    throw std::invalid_argument("Unknown layer/kernal: " + type);
  }
  const auto& spec = entries.at(type);
  for (const auto& key : {"inputs", "outputs"}) {
    const auto& ports = spec.at(key).array();
    int minimum = 0;
    int maximum = static_cast<int>(ports.size());
    for (const auto& port : ports) {
      if (!port.contains("default") && !port.contains("default_expression")) {
        ++minimum;
      }
      if (port.bool_or("variadic", false)) {
        maximum = -1;
      }
    }
    minimum = spec.int_or(std::string("min_") + key, minimum);
    maximum = spec.int_or(std::string("max_") + key, maximum);
    const auto count = std::string(key) == "inputs" ? inputs : outputs;
    if (count < size_t(minimum) || (maximum >= 0 && count > size_t(maximum))) {
      throw std::invalid_argument("Invalid input/output arity: " + type);
    }
  }
  const Json empty(Json::Object{});
  const auto& fields = spec.contains("config") ? spec.at("config") : empty;
  for (const auto& item : config.object()) {
    if (!fields.contains(item.first)) {
      throw std::invalid_argument("Unknown config field: " + type + "." +
                                  item.first);
    }
  }
  for (const auto& field : fields.object()) {
    const auto& key = field.first;
    if (!config.contains(key)) {
      if (!field.second.contains("default") &&
          !field.second.contains("default_expression")) {
        throw std::invalid_argument("Missing config field: " + type + "." +
                                    key);
      }
      continue;
    }
    const auto& value = config.at(key);
    const auto kind = field.second.at("type").string();
    if (kind == "integer") {
      value.integer();
    } else if (kind == "number") {
      value.number();
    } else if (kind == "boolean") {
      value.boolean();
    } else if (kind == "array") {
      value.array();
    } else if (kind == "object") {
      value.object();
    } else {
      value.string();
    }
    if (field.second.contains("enum")) {
      bool found = false;
      for (const auto& allowed : field.second.at("enum").array()) {
        found |= allowed.string() == value.string();
      }
      if (!found) {
        throw std::invalid_argument("Invalid enum: " + type + "." + key);
      }
    }
  }
}

std::vector<Tensor> execute_operation(const std::string& type,
                                      const std::vector<Tensor>& x,
                                      const Json& p, Backend backend,
                                      uint32_t seed, bool training) {
  if (type == "zeros" || type == "ones" || type == "normal" ||
      type == "uniform" || type == "constant") {
    const auto dims = integers(p.at("shape"));
    const auto dtype = parse_dtype(p.string_or("dtype", "float32"));
    if (type == "zeros") {
      return {Tensor::zeros(dims, backend, false, dtype)};
    }
    if (type == "ones") {
      return {Tensor::ones(dims, backend, false, dtype)};
    }
    if (type == "normal") {
      return {Tensor::normal(dims, p.number_or("mean", 0),
                             p.number_or("stddev", 1), seed, backend, false,
                             dtype)};
    }
    if (type == "uniform") {
      return {Tensor::uniform(dims, p.number_or("low", 0),
                              p.number_or("high", 1), seed, backend, false,
                              dtype)};
    }
    std::vector<double> data;
    for (const auto& value : p.at("values").array()) {
      data.push_back(value.number());
    }
    return {cast(Tensor::from_double_data(dims, data, backend), dtype)};
  }
  if (type == "reshape") {
    auto dims = integers(p.at("shape"));
    for (size_t i = 0; i < dims.size(); ++i) {
      if (!dims[i]) {
        dims[i] = x[0].shape().at(i);
      }
    }
    return {reshape(x[0], dims)};
  }
  if (type == "identity") {
    return {x.at(0)};
  }
  if (type == "embedding") {
    const auto& ids = x.at(0);
    const auto& weight = x.at(1);
    if (weight.shape().size() != 2 ||
        (ids.dtype() != DType::Int32 && ids.dtype() != DType::Int64)) {
      throw std::invalid_argument(
          "Embedding requires integer IDs and rank-two weights");
    }
    Shape dims = ids.shape();
    dims.push_back(weight.shape()[1]);
    return {reshape(index_select(weight, 0, reshape(ids, {-1})), dims)};
  }
  if (type == "flatten") {
    const auto& dims = x.at(0).shape();
    int axis = p.int_or("axis", 1);
    if (axis < 0) {
      axis += int(dims.size());
    }
    if (axis < 0 || axis > int(dims.size())) {
      throw std::invalid_argument("Flatten axis out of range");
    }
    return {reshape(
        x[0],
        {int(shape_element_count(Shape(dims.begin(), dims.begin() + axis))),
         int(shape_element_count(Shape(dims.begin() + axis, dims.end())))})};
  }
  if (type == "position_ids" || type == "causal_mask") {
    const int length = x.at(0).shape().at(p.int_or("axis", 1));
    if (type == "position_ids") {
      std::vector<int32_t> data(length);
      for (int i = 0; i < length; ++i) {
        data[i] = i;
      }
      return {Tensor::from_int_data({length}, data, backend)};
    }
    // Decoder blocks share one immutable mask per sequence length and device.
    // Reusing it avoids rebuilding and uploading the same matrix for each block.
    static thread_local std::map<std::pair<int, int>, std::map<int, Tensor>> masks;
    const auto device_key = std::make_pair(int(backend), current_device(backend));
    auto& device_masks = masks[device_key];
    if (const auto found = device_masks.find(length); found != device_masks.end())
      return {found->second};
    std::vector<float> data(size_t(length) * length);
    for (int i = 0; i < length; ++i) {
      for (int j = i + 1; j < length; ++j) {
        data[size_t(i) * length + j] = -1e9f;
      }
    }
    return {device_masks.emplace(length, Tensor::from_data({1, 1, length, length}, data, backend)).first->second};
  }
  if (type == "scaled_dot_product_attention") {
    return {scaled_dot_product_attention(x.at(0),x.at(1),x.at(2),x.size()>3?x[3]:Tensor{},
        training?p.number_or("dropout_p",0):0,p.bool_or("is_causal",p.bool_or("causal",false)),
        p.contains("scale")?std::optional<double>(p.at("scale").number()):std::nullopt,p.bool_or("enable_gqa",p.contains("causal")),seed)};
  }
  if (type == "split_heads") {
    const auto& dims = x.at(0).shape();
    const int heads = p.at("heads").integer();
    if (dims.size() != 3 || heads <= 0 || dims[2] % heads) {
      throw std::invalid_argument("Invalid attention head geometry");
    }
    return {permute(reshape(x[0], {dims[0], dims[1], heads, dims[2] / heads}),
                    {0, 2, 1, 3})};
  }
  if (type == "merge_heads") {
    const auto& dims = x.at(0).shape();
    if (dims.size() != 4) {
      throw std::invalid_argument("MergeHeads requires rank four");
    }
    return {reshape(permute(x[0], {0, 2, 1, 3}),
                    {dims[0], dims[2], dims[1] * dims[3]})};
  }
#include "Layers/KernalBindings.inc"
  throw std::invalid_argument("Missing kernal implementation: " + type);
}
}  // namespace elman::nn
