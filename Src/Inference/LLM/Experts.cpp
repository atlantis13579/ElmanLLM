#include "Inference/LLM/Experts.h"
#include "Core/Graph.h"
#include "Core/Memory.h"
#include "Core/Device.h"
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(std::string("Expert routing: ") + message);
}
}
void ExpertRoutingOptions::validate() const {
    require((score == ExpertScore::Softmax || score == ExpertScore::Sigmoid) &&
        (group_score == ExpertGroupScore::Maximum || group_score == ExpertGroupScore::TopTwo), "invalid scoring mode");
    require(experts > 0 && top_k > 0 && top_k <= experts, "invalid expert count or top-k");
    require(groups > 0 && experts % groups == 0 && selected_groups > 0 && selected_groups <= groups,
            "invalid expert groups");
    require(int64_t(selected_groups) * (experts / groups) >= top_k, "selected groups cannot supply top-k experts");
    require(group_score != ExpertGroupScore::TopTwo || experts / groups >= 2, "top-two groups require two experts");
    require(std::isfinite(scale) && scale > 0 && std::isfinite(normalization_epsilon) && normalization_epsilon >= 0,
            "invalid mixture scaling or normalization");
}

ExpertRoutes route_experts(const Tensor& logits, const ExpertRoutingOptions& options, const Tensor& correction_bias) {
    options.validate();
    require(logits.defined() && logits.shape().size() == 2 && logits.shape()[1] == options.experts,
            "logits must have shape [tokens, experts]");
    require(is_compute_float(logits.dtype()) && !logits.requires_grad(), "routing requires inference floating logits");
    DeviceScope owner(logits.backend(), logits.device());
    const int tokens = logits.shape()[0];
    graph::RecordingGuard inference;
    auto wide = cast(logits, DType::Float32);
    Tensor scores;
    if (options.score == ExpertScore::Softmax) scores = softmax(wide, -1);
    else scores = div(Tensor::ones(logits.shape(), logits.backend()), add(exp(mul(wide, -1.)), 1.));
    auto selection = scores;
    if (correction_bias.defined()) {
        require(correction_bias.shape() == Shape{options.experts} && correction_bias.backend() == logits.backend() && correction_bias.device() == logits.device() &&
                is_compute_float(correction_bias.dtype()) && !correction_bias.requires_grad(), "invalid correction bias");
        selection = add(selection, cast(correction_bias, DType::Float32));
    }
    if (options.groups > 1 && options.selected_groups < options.groups) {
        const int width = options.experts / options.groups;
        auto grouped = reshape(selection, {tokens, options.groups, width});
        Tensor group_scores;
        if (options.group_score == ExpertGroupScore::Maximum) group_scores = amax(grouped, -1);
        else {
            auto best = slice(argsort(grouped, -1, true), -1, 0, 2);
            group_scores = sum(gather(grouped, -1, best), -1);
        }
        auto chosen = slice(argsort(group_scores, -1, true), -1, 0, options.selected_groups);
        auto mask = scatter(Tensor::zeros({tokens, options.groups}, logits.backend()), 1, chosen,
                            Tensor::ones({tokens, options.selected_groups}, logits.backend()));
        mask = reshape(expand(unsqueeze(mask, -1), {tokens, options.groups, width}), logits.shape());
        auto excluded = Tensor::zeros(logits.shape(), logits.backend());
        if (!options.zero_masked_groups) excluded = add(excluded, -std::numeric_limits<float>::infinity());
        selection = where(cast(mask, DType::Bool), selection, excluded);
    }
    auto indices = slice(argsort(selection, -1, true), -1, 0, options.top_k);
    auto weights = gather(scores, -1, indices);
    if (options.normalize) weights = div(weights, add(sum(weights, -1, true), options.normalization_epsilon));
    if (!options.normalize || options.scale_normalized) weights = mul(weights, options.scale);
    return {indices, weights, options.experts};
}

Tensor evaluate_experts(const Tensor& input, const ExpertRoutes& routes,
                        const std::function<Tensor(int, const Tensor&)>& expert) {
    require(input.defined() && input.shape().size() == 2 && is_compute_float(input.dtype()) && !input.requires_grad(),
            "expert input must be an inference floating matrix");
    DeviceScope owner(input.backend(), input.device());
    require(bool(expert) && routes.experts > 0 && routes.indices.defined() && routes.weights.defined(), "missing routing data");
    require(routes.indices.shape().size() == 2 && routes.indices.shape()[0] == input.shape()[0] &&
            routes.indices.shape()[1] > 0 && routes.indices.shape()[1] <= routes.experts &&
            routes.weights.shape() == routes.indices.shape(), "route geometry differs");
    require(routes.indices.dtype() == DType::Int32 && routes.weights.dtype() == DType::Float32 &&
            routes.indices.backend() == input.backend() && routes.weights.backend() == input.backend() &&
            routes.indices.device() == input.device() && routes.weights.device() == input.device() &&
            !routes.weights.requires_grad(), "route backend or dtype differs");
    const int tokens = input.shape()[0], hidden = input.shape()[1], top_k = routes.indices.shape()[1];
    require(int64_t(tokens) * top_k <= INT_MAX && int64_t(tokens) * top_k * hidden <= INT_MAX,
            "expert output exceeds indexing limits");
    graph::RecordingGuard inference;
    MemoryReservation metadata(Backend::CPU, checked_memory_add(checked_memory_mul(routes.indices.numel(), sizeof(int32_t) * 5),
        checked_memory_mul(size_t(routes.experts), sizeof(std::vector<int32_t>) * 2)));
    auto ids = routes.indices.int_data();
    std::vector<std::vector<int32_t>> token_rows(routes.experts), destination_rows(routes.experts);
    for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < top_k; ++slot) {
        const int destination = token * top_k + slot, id = ids[destination];
        require(id >= 0 && id < routes.experts, "expert index is outside the configured range");
        for (int previous = 0; previous < slot; ++previous)
            require(ids[token * top_k + previous] != id, "an expert occurs twice for one token");
        token_rows[id].push_back(token);
        destination_rows[id].push_back(destination);
    }
    auto output = Tensor::zeros({tokens * top_k, hidden}, input.backend(), false, input.dtype());
    for (int id = 0; id < routes.experts; ++id) {
        const int count = int(token_rows[id].size());
        if (!count) continue;
        auto rows = Tensor::from_int_data({count}, token_rows[id], input.backend());
        auto values = expert(id, index_select(input, 0, rows));
        require(values.defined() && values.shape() == Shape{count, hidden} && values.backend() == input.backend() &&
                values.device() == input.device() &&
                values.dtype() == input.dtype() && !values.requires_grad(), "expert returned an incompatible output");
        auto destinations = Tensor::from_int_data({count}, destination_rows[id], input.backend());
        output.assign_indexed(0, destinations, values);
    }
    auto grouped = reshape(cast(output, DType::Float32), {tokens, top_k, hidden});
    return cast(sum(mul(grouped, unsqueeze(routes.weights, -1)), 1), input.dtype());
}
} // namespace elman
