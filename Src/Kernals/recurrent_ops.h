#pragma once

#include "Core/Tensor.h"

namespace elman {

struct RecurrentOutput { Tensor values, state; };
Tensor sigmoid(const Tensor& input);

// Input is [batch, time, channels], weights are [channels, kernel].
// State stores the preceding kernel - 1 inputs in chronological order.
RecurrentOutput causal_depthwise_conv1d(const Tensor& input, const Tensor& weight, const Tensor& state = {});

// Query/key: [batch, time, key heads, key width]; value: [batch, time, value heads, value width].
// Decay is logarithmic; beta contains update gates. State is Float32 [batch, value heads, key width, value width].
// Q/K L2 normalization and the inverse square-root query scale are included.
RecurrentOutput gated_delta_rule(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& log_decay, const Tensor& beta, const Tensor& state = {});

} // namespace elman
