#pragma once

#include <string>
#include <utility>
#include <vector>

#include "Core/Tensor.h"
#include "Training/Sampler.h"

namespace elman {

using TensorBatch = std::pair<Tensor, Tensor>;

// Dataset implementations can read from an indexed file, generate rows on
// demand, or keep rows in memory. The fingerprint identifies immutable data
// and transforms for all ranks and checkpoint resumes.
class BatchDataset {
 public:
  virtual ~BatchDataset() = default;
  virtual size_t size() const = 0;
  virtual const Shape& shape() const = 0;
  virtual std::string fingerprint() const = 0;
  virtual TensorBatch batch(const std::vector<size_t>& order, size_t offset,
                            size_t count, Backend backend) const = 0;
  // Loss is the mean across samples in the supplied batch.
  virtual Tensor loss(const Tensor& prediction, const Tensor& target) const = 0;
};

class DataLoader {
 public:
  DataLoader(const BatchDataset& dataset, Backend backend) : dataset_(dataset), backend_(backend) {}
  TensorBatch batch(const std::vector<size_t>& order, size_t offset, size_t count) const;
  TensorBatch next(BatchSampler& sampler, size_t global_count, size_t offset, size_t count) const;

 private:
  const BatchDataset& dataset_;
  Backend backend_;
};

}  // namespace elman
