#include "Runtime/DiffusionCommand.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Core/Device.h"
#include "Inference/Diffusion/StableDiffusion.h"
#include "IO/AtomicFile.h"
#include "IO/Jpeg.h"
#include "IO/Png.h"
#include "IO/Safetensors.h"

namespace elman {
namespace {

template<class T> T integer(const std::string& option, const std::string& text) {
  T result{};
  const char* const end = text.data() + text.size();
  const std::from_chars_result parsed = std::from_chars(text.data(), end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end) throw std::invalid_argument("Invalid integer for " + option);
  return result;
}

double real(const std::string& option, const std::string& text) {
  size_t end = 0;
  const double result = std::stod(text, &end);
  if (end != text.size() || !std::isfinite(result)) throw std::invalid_argument("Invalid number for " + option);
  return result;
}

}  // namespace

int diffusion_command(int argc, char** argv) {
  if (argc < 3) throw std::invalid_argument("diffusion requires a model directory");
  const std::filesystem::path root(argv[2]);
  std::string prompt, negative, output, requested_dtype, latent_path, trace_directory;
  bool has_prompt = false;
  Backend backend = Backend::CPU;
  uint32_t seed = 1;
  int width = 0, height = 0, steps = 20;
  double guidance = 7.5;
  for (int i = 3; i < argc; ++i) {
    const std::string option = argv[i];
    if (i + 1 == argc) throw std::invalid_argument("Missing value for " + option);
    const std::string value = argv[++i];
    if (option == "--prompt") { prompt = value; has_prompt = true; }
    else if (option == "--negative-prompt") negative = value;
    else if (option == "--output") output = value;
    else if (option == "--backend") backend = parse_backend(value);
    else if (option == "--dtype") requested_dtype = value;
    else if (option == "--width") width = integer<int>(option, value);
    else if (option == "--height") height = integer<int>(option, value);
    else if (option == "--steps") steps = integer<int>(option, value);
    else if (option == "--seed") seed = integer<uint32_t>(option, value);
    else if (option == "--guidance") guidance = real(option, value);
    else if (option == "--initial-latent") latent_path = value;
    else if (option == "--trace-dir") trace_directory = value;
    else throw std::invalid_argument("Unknown diffusion option " + option);
  }
  std::string output_type = std::filesystem::path(output).extension().string();
  std::transform(output_type.begin(), output_type.end(), output_type.begin(), [](unsigned char c) { return std::tolower(c); });
  const bool jpeg = output_type == ".jpg" || output_type == ".jpeg";
  if (!has_prompt || (!jpeg && output_type != ".png" && output_type != ".safetensors") || width < 0 || height < 0 || guidance < 0)
    throw std::invalid_argument("diffusion requires --prompt and --output ending in .png, .jpg, .jpeg or .safetensors");
  const DType dtype = requested_dtype.empty() ? (backend == Backend::CUDA ? DType::Float16 : DType::Float32)
                                             : parse_dtype(requested_dtype);
  if (!is_compute_float(dtype)) throw std::invalid_argument("diffusion requires float32, float16 or bfloat16");
  const int device = current_device(backend);
  std::cerr << "[sd] Loading model " << root.string() << " (" << backend_name(backend) << ", " << dtype_name(dtype) << ")\n";
  DiffusionSession session = DiffusionSession::load(root, backend, device, dtype);
  DiffusionInput input{prompt, negative, width, height, steps, seed, guidance, {}};
  if (!latent_path.empty()) input.initial_latent = SafetensorsFile(latent_path).load("latent", backend);
  DiffusionTrace trace;
  if (!trace_directory.empty()) {
    trace = [&](const std::string& name, const Tensor& tensor) {
      std::filesystem::path path = std::filesystem::path(trace_directory) / (name + ".safetensors");
      save_safetensors(path.string(), {{name, tensor}});
    };
  }
  const StableDiffusionOutput result = session.generate(input, trace, [](const DiffusionProgress& progress) {
    std::cerr << "[sd] " << diffusion_progress_text(progress) << '\n';
  });
  std::cerr << "[sd] Writing " << output << '\n';
  if (jpeg || output_type == ".png") {
    write_file_atomically(output, [&](const std::filesystem::path& temporary) {
      std::ofstream file(temporary, std::ios::binary);
      if (!file) throw std::runtime_error("Cannot create image file");
      file << (jpeg ? encode_jpeg(result.image) : encode_png(result.image));
      file.close();
      if (!file) throw std::runtime_error("Cannot close image file");
    });
  } else save_safetensors(output, {{"image", result.image}, {"latent", result.latent}});
  std::cerr << "[sd] Completed\n";
  return 0;
}

}  // namespace elman
