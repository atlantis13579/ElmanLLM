#include "Reinforcement/Reinforcement.h"
#include <cmath>

namespace elman::rl {

std::vector<float> discounted_returns(const std::vector<float>& rewards, double discount, double bootstrap) {
    if (!(discount >= 0 && discount <= 1) || !std::isfinite(bootstrap))
        throw std::invalid_argument("Invalid discount or bootstrap value");
    std::vector<float> result(rewards.size());
    for (size_t i = rewards.size(); i-- > 0;) {
        bootstrap = rewards[i] + discount * bootstrap;
        result[i] = static_cast<float>(bootstrap);
        if (!std::isfinite(result[i])) throw std::invalid_argument("Non-finite discounted return");
    }
    return result;
}

Tensor policy_gradient_loss(const Tensor& logits, const Tensor& actions, const Tensor& advantages) {
    if (logits.shape().size() != 2 || logits.shape()[0] <= 0 || logits.shape()[1] <= 0 ||
        actions.shape() != Shape{logits.shape()[0]} || advantages.shape() != actions.shape() ||
        advantages.dtype() != logits.dtype() || actions.backend() != logits.backend() ||
        advantages.backend() != logits.backend() || (actions.dtype() != DType::Int32 && actions.dtype() != DType::Int64))
        throw std::invalid_argument("Expected categorical logits [N,A], integer actions [N] and advantages [N]");
    Tensor indices = actions;
    if (actions.dtype() == DType::Int64) {
        std::vector<int32_t> narrow;
        for (int64_t action : actions.int64_data()) {
            if (action < 0 || action >= logits.shape()[1]) throw std::out_of_range("Action outside policy");
            narrow.push_back(static_cast<int32_t>(action));
        }
        indices = Tensor::from_int_data(actions.shape(), narrow, actions.backend());
    }
    Tensor selected = gather(log_softmax(logits, 1), 1, unsqueeze(indices, 1));
    return mul(mean_all(mul(squeeze(selected, 1), advantages.detach())), -1);
}

} // namespace elman::rl
