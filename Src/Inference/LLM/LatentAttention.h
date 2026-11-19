#pragma once

#include "Core/Tensor.h"

namespace elman {
struct LatentKVCache {
    Tensor latent; // [batch, tokens, rank], normalized compressed K/V.
    Tensor rotary; // [batch, tokens, rotary_dimension], already rotated.
    int length() const;
};

// Immutable append: a failed forward never changes a caller's committed cache.
LatentKVCache append_latent_cache(const LatentKVCache* previous, const Tensor& latent, const Tensor& rotary);

// Weight absorption computes scores and values without expanding cached K/V
// to a tensor per attention head. Query operands are [batch, heads, queries, D].
// Key/value projection weights are [heads, D, rank]. Padding is additive and
// broadcastable to [batch, heads, queries, keys]. Softmax uses Float32.
Tensor latent_attention(const Tensor& query_plain, const Tensor& query_rotary,
                        const LatentKVCache& cache, const Tensor& key_weight,
                        const Tensor& value_weight, const Tensor& padding,
                        double scale, int causal_offset, int query_tile = 128);
} // namespace elman
