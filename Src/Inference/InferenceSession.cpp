#include "Inference/InferenceSession.h"
#include "Core/LocalFunction.h"
#include "Inference/GraphSession.h"
#include "Inference/Importers/Keras.h"
#include "Inference/Importers/SavedModel.h"
#include "Inference/Importers/TfLite.h"
#include "Inference/Importers/TorchScript.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "Core/Device.h"
#include "Core/Graph.h"
#include "IO/Checkpoint.h"
#include "IO/Safetensors.h"
#include "IO/TensorArchive.h"

namespace elman {
namespace {

void validate_signature(const TensorSignature& signature) {
  if (signature.empty()) throw std::invalid_argument("Inference requires named tensor ports");
  for (const TensorSignature::value_type& entry : signature) {
    const std::string& name = entry.first;
    const TensorSpec& spec = entry.second;
    if (!spec.any_dtype) dtype_info(spec.dtype);
    if (name.empty() || spec.shape.size() > 8 ||
        std::any_of(spec.shape.begin(), spec.shape.end(), [](int dim) { return dim < -1; }))
      throw std::invalid_argument("Invalid inference tensor signature for " + name);
  }
}

void validate_tensors(const NamedTensors& tensors, const TensorSignature& signature,
                      Backend backend, int device, const char* role) {
  if (tensors.size() != signature.size())
    throw std::invalid_argument(std::string("Unexpected inference ") + role + " count");
  for (const TensorSignature::value_type& entry : signature) {
    const std::string& name = entry.first;
    const TensorSpec& spec = entry.second;
    auto found = tensors.find(name);
    if (found == tensors.end() || !found->second.defined())
      throw std::invalid_argument(std::string("Missing inference ") + role + " " + name);
    const Tensor& value = found->second;
    if ((!spec.any_dtype && value.dtype() != spec.dtype) || value.backend() != backend || value.device() != device ||
        (!spec.any_rank && value.shape().size() != spec.shape.size())) {
      throw std::invalid_argument(std::string("Invalid inference ") + role + " dtype, device or rank for " + name);
    }
    for (size_t axis = 0; !spec.any_rank && axis < spec.shape.size(); ++axis) {
      if (spec.shape[axis] != -1 && spec.shape[axis] != value.shape()[axis])
        throw std::invalid_argument(std::string("Invalid inference ") + role + " shape for " + name);
    }
  }
}

}  // namespace

InferenceSession::InferenceSession(Backend backend, int device, TensorSignature inputs,
                                   TensorSignature outputs, Forward forward)
    : backend_(backend), device_(device), inputs_(std::move(inputs)),
      outputs_(std::move(outputs)), forward_(std::move(forward)) {
  if (!forward_) throw std::invalid_argument("Inference session requires a forward function");
  validate_signature(inputs_);
  validate_signature(outputs_);
}

NamedTensors InferenceSession::run_with_grad(const NamedTensors& inputs) const {
  validate_tensors(inputs, inputs_, backend_, device_, "input");
  DeviceScope owner(backend_, device_);
  auto outputs = forward_(inputs);
  validate_tensors(outputs, outputs_, backend_, device_, "output");
  return outputs;
}

NamedTensors InferenceSession::run(const NamedTensors& inputs) const {
  validate_tensors(inputs, inputs_, backend_, device_, "input");
  DeviceScope owner(backend_, device_);
  graph::RecordingGuard no_grad;
  std::map<std::string, Tensor> outputs = forward_(inputs);
  validate_tensors(outputs, outputs_, backend_, device_, "output");
  for (NamedTensors::value_type& entry : outputs) {
    Tensor& output = entry.second;
    if (output.requires_grad()) output = output.detach();
  }
  return outputs;
}

void load_inference_parameters(const std::filesystem::path& path, Model& model, const Json& definition,
                               Backend backend) {
  if (!is_tensor_archive(path)) {
    load_checkpoint(path.string(), model, definition);
    return;
  }
  const std::vector<Parameter> buffers = model.checkpoint_buffers();
  const std::vector<Parameter>& parameters = model.parameters();
  const TensorArchive tensors = load_tensor_archive(path, backend, parameters.size() == 1 ? parameters.front().name : "input");
  if (tensors.size() != parameters.size() + buffers.size())
    throw std::invalid_argument("Inference weights count does not match model parameters and buffers");
  LocalFunction check{[&](const Parameter& target) {
    const auto found = tensors.find(target.name);
    if (found == tensors.end() || found->second.dtype() != target.value.dtype() || found->second.shape() != target.value.shape())
      throw std::invalid_argument("Missing or incompatible inference weight " + target.name);
  }};
  for (const Parameter& target : parameters) check(target);
  for (const Parameter& target : buffers) check(target);
  for (Parameter& target : model.parameters()) target.value.assign_raw(tensors.at(target.name).raw_data());
  for (Parameter target : buffers) target.value.assign_raw(tensors.at(target.name).raw_data());
}

InferenceSession load_model_session(const std::string& model_path, const std::string& checkpoint_path,
                                    Backend backend, int device) {
  DeviceScope owner(backend, device);
  const Json definition = Json::read(model_path);
  std::shared_ptr<Model> model = create_model(definition, backend);
  load_inference_parameters(std::filesystem::u8path(checkpoint_path), *model, definition, backend);
  Shape shape = model->sample_shape();
  shape.insert(shape.begin(), -1);
  const bool language = model->architecture() == "gpt2";
  if (language) shape.back() = -1;
  const Shape output = language ? Shape{-1, -1, model->output_classes()} : Shape{-1, model->output_classes()};
  return InferenceSession(backend, device,
      {{"input", {language ? DType::Int32 : DType::Float32, std::move(shape)}}},
      {{"output", {DType::Float32, output}}}, [model = std::move(model)](const NamedTensors& inputs) {
        return NamedTensors{{"output", model->forward(inputs.at("input"), false)}};
      });
}

int infer_tensors_command(int argc, char** argv) {
  if (argc < 3) throw std::invalid_argument("infer-tensors requires a model path");
  std::string input_path, output_path, checkpoint, signature;
  Backend backend = Backend::CPU;
  for (int i = 3; i < argc; ++i) {
    const std::string option = argv[i];
    if (i + 1 == argc) throw std::invalid_argument("Missing value for " + option);
    const std::string value = argv[++i];
    if (option == "--input") input_path = value;
    else if (option == "--output") output_path = value;
    else if (option == "--checkpoint") checkpoint = value;
    else if (option == "--signature") signature = value;
    else if (option == "--backend") backend = parse_backend(value);
    else throw std::invalid_argument("Unknown infer-tensors argument " + option);
  }
  if (input_path.empty() || output_path.empty())
    throw std::invalid_argument("infer-tensors requires --input and --output");
  const std::filesystem::path model = std::filesystem::u8path(argv[2]);
  const std::string extension = model_file_extension(model);
  const bool tflite = extension == ".tflite";
  const bool keras = extension == ".keras" || extension == ".h5" || extension == ".hdf5";
  const bool scripted = extension == ".pt" || extension == ".pth" || extension == ".torchscript" || extension == ".ptc";
  const bool saved = extension == ".pb" || std::filesystem::is_directory(model);
  const bool graph = tflite || keras || saved || scripted || Json::read(model).at("format").string() == "elman.graph";
  if (!saved && !signature.empty()) throw std::invalid_argument("--signature requires a TensorFlow SavedModel");
  if (!graph && checkpoint.empty())
    throw std::invalid_argument("Built-in models require --checkpoint");
  if (graph && !checkpoint.empty())
    throw std::invalid_argument("Graph models specify their own initializers");
  const int device = current_device(backend);
  InferenceSession session = tflite ? load_tflite_session(model, backend, device) :
                 keras ? load_keras_session(model, backend, device) :
                 scripted ? load_torchscript_session(model, backend, device) :
                 saved ? load_saved_model_session(model, backend, device, signature.empty() ? "serving_default" : signature) :
                 graph ? load_graph_session(model, backend, device) :
                         load_model_session(argv[2], checkpoint, backend, device);
  const std::string name = session.inputs().size() == 1 ? session.inputs().begin()->first : "input";
  DeviceScope owner(backend, device);
  const TensorArchive tensors = load_tensor_archive(std::filesystem::u8path(input_path), backend, name);
  save_safetensors(std::filesystem::u8path(output_path), session.run(tensors));
  return 0;
}

}  // namespace elman
