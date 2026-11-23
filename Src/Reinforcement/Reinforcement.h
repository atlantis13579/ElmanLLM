#pragma once

#include "Core/Tensor.h"
#include <random>
#include <stdexcept>

namespace elman::rl {

// Rewards are chronological. Bootstrap is the value after the last reward.
std::vector<float> discounted_returns(const std::vector<float>& rewards, double discount,
                                      double bootstrap = 0);

// One categorical action and detached advantage per row of logits.
Tensor policy_gradient_loss(const Tensor& logits, const Tensor& actions, const Tensor& advantages);

template <class Sample>
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity) : capacity_(capacity) {
        if (!capacity) throw std::invalid_argument("Replay capacity must be positive");
    }
    void push(Sample sample) {
        if (data_.size() < capacity_) data_.push_back(std::move(sample));
        else data_[next_] = std::move(sample);
        next_ = (next_ + 1) % capacity_;
    }
    size_t size() const { return data_.size(); }

    // Uniform sampling with replacement. The caller owns the RNG state.
    const Sample& draw(std::mt19937& random) const {
        if (data_.empty()) throw std::logic_error("Cannot draw from empty replay");
        return data_[std::uniform_int_distribution<size_t>(0, data_.size() - 1)(random)];
    }
private:
    size_t capacity_, next_ = 0;
    std::vector<Sample> data_;
};

} // namespace elman::rl
