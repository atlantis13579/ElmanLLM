#include "Inference/InferenceSession.h"
#include "Inference/GraphSession.h"
#include "Inference/Importers/Keras.h"
#include "Inference/Importers/SavedModel.h"
#include "Inference/Importers/TfLite.h"
#include "Inference/Importers/TorchScript.h"

#include <stdexcept>

#include "Core/Device.h"
#include "IO/Safetensors.h"
#include "IO/Json.h"
#include "IO/TensorArchive.h"

#include "Commands/Commands.h"

namespace elman {
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

} // namespace elman
