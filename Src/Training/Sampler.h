#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace elman {

struct SamplerState {
  std::vector<size_t> order;
  size_t offset = 0;
  uint32_t random_state = 1;
};

class BatchSampler {
 public:
  BatchSampler(size_t samples, uint32_t seed, bool shuffle);
  std::vector<size_t> next(size_t count);
  const SamplerState& state() const { return state_; }
  void restore(SamplerState state);

 private:
  SamplerState state_;
  bool shuffle_;
};

}  // namespace elman
