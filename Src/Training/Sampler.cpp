#include "Training/Sampler.h"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace elman {
namespace {

uint32_t next_random(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

}  // namespace

BatchSampler::BatchSampler(size_t samples, uint32_t seed, bool shuffle)
    : shuffle_(shuffle) {
  if (!samples) throw std::invalid_argument("Sampler requires a nonempty dataset");
  state_.order.resize(samples);
  std::iota(state_.order.begin(), state_.order.end(), size_t{0});
  state_.random_state = seed ? seed : 1;
}

std::vector<size_t> BatchSampler::next(size_t count) {
  std::vector<size_t> selected;
  selected.reserve(count);
  auto& order = state_.order;
  for (size_t sample = 0; sample < count; ++sample) {
    if (state_.offset == 0 && shuffle_) {
      for (size_t i = order.size(); i > 1; --i) {
        std::swap(order[i - 1], order[next_random(state_.random_state) % i]);
      }
    }
    selected.push_back(order[state_.offset]);
    state_.offset = (state_.offset + 1) % order.size();
  }
  return selected;
}

void BatchSampler::restore(SamplerState state) {
  if (state.order.size() != state_.order.size() ||
      state.offset >= state.order.size() || state.random_state == 0) {
    throw std::invalid_argument("Sampler checkpoint does not match the dataset");
  }
  std::vector<bool> seen(state.order.size(), false);
  for (size_t i = 0; i < state.order.size(); ++i) {
    size_t index = state.order[i];
    if (index >= seen.size() || seen[index] || (!shuffle_ && index != i)) {
      throw std::invalid_argument("Invalid sample order in checkpoint");
    }
    seen[index] = true;
  }
  state_ = std::move(state);
}

}  // namespace elman
