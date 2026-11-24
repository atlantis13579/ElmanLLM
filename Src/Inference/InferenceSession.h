#pragma once

#include <filesystem>

#include <functional>
#include <map>
#include <string>

#include "Core/Tensor.h"

namespace elman {

class Json;
class Model;

using NamedTensors = std::map<std::string, Tensor>;

struct TensorSpec {
  DType dtype = DType::Float32;
  Shape shape;  // A dimension of -1 accepts any runtime size.
  bool any_rank = false;
  bool any_dtype = false;
};

using TensorSignature = std::map<std::string, TensorSpec>;

class InferenceSession {
 public:
  using Forward = std::function<NamedTensors(const NamedTensors&)>;

  InferenceSession(Backend backend, int device, TensorSignature inputs,
                   TensorSignature outputs, Forward forward);

  Backend backend() const { return backend_; }
  int device() const { return device_; }
  const TensorSignature& inputs() const { return inputs_; }
  const TensorSignature& outputs() const { return outputs_; }
  NamedTensors run(const NamedTensors& inputs) const;
  // Explicit differentiable execution for declarative training graphs.
  NamedTensors run_with_grad(const NamedTensors& inputs) const;

 private:
  Backend backend_;
  int device_;
  TensorSignature inputs_;
  TensorSignature outputs_;
  Forward forward_;
};

InferenceSession load_model_session(const std::string& model_path, const std::string& checkpoint_path,
                                    Backend backend, int device);
void load_inference_parameters(const std::filesystem::path& path, Model& model, const Json& definition,
                               Backend backend);

}  // namespace elman
