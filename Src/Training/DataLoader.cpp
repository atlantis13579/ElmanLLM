#include "Training/DataLoader.h"

#include <stdexcept>

namespace elman {

TensorBatch DataLoader::batch(const std::vector<size_t>& order, size_t offset, size_t count) const {
  if (!count || offset > order.size() || count > order.size() - offset)
    throw std::invalid_argument("Invalid data loader batch range");
  return dataset_.batch(order, offset, count, backend_);
}

TensorBatch DataLoader::next(BatchSampler& sampler, size_t global_count, size_t offset, size_t count) const {
  return batch(sampler.next(global_count), offset, count);
}

}  // namespace elman
