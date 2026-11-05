#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Models/Model.h"
#include "Runtime/RunOptions.h"

namespace elman {

struct Configuration {
  Json json;
  Json definition;
  std::filesystem::path directory;
  Backend backend;
  uint32_t seed;
  std::unique_ptr<Model> model;

  explicit Configuration(const std::string& path, const RunOptions& options);
  std::string output(const std::string& key, const std::string& fallback) const;
};

struct Dataset {
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> targets;
  Shape sample_shape;
  bool language = false;
  int classes = 0;

  explicit Dataset(const Configuration& config);
  std::pair<Tensor, Tensor> batch(const std::vector<size_t>& order,
                                  size_t offset, size_t count,
                                  Backend backend) const;
};

Tensor classification_loss(const Tensor& logits, const Tensor& targets,
                           int classes);

}  // namespace elman
