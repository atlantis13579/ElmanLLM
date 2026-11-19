#include "Runtime/RunContext.h"

#include "Core/Device.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace elman {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string resolve(const std::filesystem::path& base,
                    const std::string& path) {
  auto result = std::filesystem::path(path);
  if (result.is_relative()) {
    result = base / result;
  }
  return std::filesystem::absolute(result).lexically_normal().string();
}

}  // namespace

Configuration::Configuration(const std::string& path,
                             const RunOptions& options, int cuda_device) {
  directory = std::filesystem::absolute(path).parent_path();
  json = Json::read(path);
  require(json.at("format").string() == "elman.training" &&
              json.at("version").integer() == 1,
          "Unsupported training configuration format/version");
  definition = Json::read(resolve(directory, json.at("model").string()));
  backend =
      parse_backend(options.backend.empty() ? json.string_or("backend", "cpu")
                                            : options.backend);
  require(backend_available(backend),
          "Requested backend is unavailable: " + backend_name(backend));
  int seed_value = json.at("training").int_or("seed", 42);
  require(seed_value >= 0, "Seed must be non-negative");
  seed = static_cast<uint32_t>(seed_value);
  DeviceScope device(backend, backend == Backend::CUDA && cuda_device >= 0
                                  ? cuda_device : current_device(backend));
  model = create_model(definition, backend, seed);
  require(json.at("model_name").string() == model->name(),
          "Training model_name does not match model definition");
}

std::string Configuration::output(const std::string& key,
                                  const std::string& fallback) const {
  return resolve(directory, json.contains("output")
                                ? json.at("output").string_or(key, fallback)
                                : fallback);
}

Dataset::Dataset(const Configuration& config) {
  const auto& settings = config.json.at("data");
  sample_shape = config.model->sample_shape();
  language = config.model->architecture() == "gpt2";
  regression = config.model->regression();
  if (language && settings.contains("sequence_length")) {
    int length = settings.at("sequence_length").integer();
    require(length > 0 && length <= sample_shape[0],
            "Dataset sequence_length exceeds the model context");
    sample_shape[0] = length;
  }
  classes = config.model->output_classes();
  const std::string format =
      regression                                              ? "regression_csv"
      : language                                              ? "token_csv"
      : config.model->architecture() == "logistic_regression" ? "tabular_csv"
                                                              : "image_csv";
  require(settings.at("format").string() == format,
          "Dataset format does not match model architecture");
  std::string path = resolve(config.directory, settings.at("path").string());
  std::ifstream input(path);
  require(static_cast<bool>(input), "Cannot open dataset: " + path);
  int limit = settings.int_or("limit_samples", 0);
  require(limit >= 0, "limit_samples must be non-negative");
  size_t sample_size = 1;
  for (int dimension : sample_shape) {
    sample_size *= static_cast<size_t>(dimension);
  }

  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }
    std::vector<float> row;
    size_t begin = 0;
    while (begin <= line.size()) {
      size_t comma = line.find(',', begin);
      std::string field = line.substr(
          begin, comma == std::string::npos ? comma : comma - begin);
      size_t consumed = 0;
      float value;
      try {
        value = std::stof(field, &consumed);
      } catch (const std::exception&) {
        throw std::runtime_error("Invalid dataset number at line " +
                                 std::to_string(line_number));
      }
      require(field.find_first_not_of(" \t\r", consumed) == std::string::npos &&
                  std::isfinite(value),
              "Invalid dataset number at line " + std::to_string(line_number));
      row.push_back(value);
      if (comma == std::string::npos) {
        break;
      }
      begin = comma + 1;
    }
    const size_t target_columns = regression ? size_t(classes) : 1;
    require(row.size() == sample_size + target_columns,
            "Dataset line " + std::to_string(line_number) + " requires " +
                std::to_string(sample_size + target_columns) + " values");
    if (language) {
      for (float token : row) {
        require(token >= 0 && token < classes && std::floor(token) == token,
                "Token ID outside vocabulary at dataset line " +
                    std::to_string(line_number));
      }
      inputs.emplace_back(row.begin(), row.end() - 1);
      targets.emplace_back(row.begin() + 1, row.end());
    } else if (regression) {
      targets.emplace_back(row.begin(), row.begin() + classes);
      inputs.emplace_back(row.begin() + classes, row.end());
    } else {
      require(row[0] >= 0 && row[0] < classes && std::floor(row[0]) == row[0],
              "Invalid target class at dataset line " +
                  std::to_string(line_number));
      inputs.emplace_back(row.begin() + 1, row.end());
      targets.push_back({row[0]});
    }
    if (limit && inputs.size() >= static_cast<size_t>(limit)) {
      break;
    }
  }
  require(!inputs.empty(), "Dataset contains no samples");
  require(input.eof() || static_cast<bool>(input), "Dataset read failed");
}

std::string Dataset::fingerprint() const {
  uint64_t hash = 14695981039346656037ull;
  for (const auto* rows : {&inputs, &targets}) {
    for (const auto& row : *rows) {
      for (float value : row) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
          hash = (hash ^ uint8_t(bits >> shift)) * 1099511628211ull;
        }
      }
    }
  }
  return std::to_string(hash);
}

TensorBatch Dataset::batch(const std::vector<size_t>& order,
                                         size_t offset, size_t count,
                                         Backend backend) const {
  std::vector<float> input_values;
  std::vector<float> target_values;
  input_values.reserve(count * inputs[0].size());
  target_values.reserve(count * targets[0].size());
  for (size_t i = 0; i < count; ++i) {
    size_t index = order[(offset + i) % order.size()];
    input_values.insert(input_values.end(), inputs[index].begin(),
                        inputs[index].end());
    target_values.insert(target_values.end(), targets[index].begin(),
                         targets[index].end());
  }
  Shape shape = sample_shape;
  shape.insert(shape.begin(), static_cast<int>(count));
  return {Tensor::from_data(shape, input_values, backend, false,
                            language ? DType::Int32 : DType::Float32),
          Tensor::from_data(regression
                                ? Shape{static_cast<int>(count), classes}
                                : Shape{static_cast<int>(target_values.size())},
                            target_values, backend, false,
                            regression ? DType::Float32 : DType::Int32)};
}

Tensor Dataset::loss(const Tensor& prediction, const Tensor& target) const {
  return regression ? mse_loss(prediction, target)
                    : classification_loss(prediction, target, classes);
}

Tensor classification_loss(const Tensor& logits, const Tensor& targets,
                           int classes) {
  return cross_entropy(
      reshape(logits, {static_cast<int>(logits.numel() / classes), classes}),
      targets);
}

}  // namespace elman
