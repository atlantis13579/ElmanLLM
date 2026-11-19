#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Models/Model.h"
#include "Runtime/RunOptions.h"
#include "Training/DataLoader.h"

namespace elman {

struct Configuration {
  Json json;
  Json definition;
  std::filesystem::path directory;
  Backend backend;
  uint32_t seed;
  std::unique_ptr<Model> model;

  explicit Configuration(const std::string& path, const RunOptions& options,
                         int cuda_device = -1);
  std::string output(const std::string& key, const std::string& fallback) const;
};

struct Dataset final : BatchDataset {
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> targets;
  Shape sample_shape;
  bool language = false;
  bool regression = false;
  int classes = 0;

  explicit Dataset(const Configuration& config);
  size_t size() const override { return inputs.size(); }
  const Shape& shape() const override { return sample_shape; }
  std::string fingerprint() const override;
  TensorBatch batch(const std::vector<size_t>& order, size_t offset,
                    size_t count, Backend backend) const override;
  Tensor loss(const Tensor& prediction, const Tensor& target) const override;
};

Tensor classification_loss(const Tensor& logits, const Tensor& targets,
                           int classes);

}  // namespace elman
