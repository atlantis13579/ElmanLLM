#pragma once

#include "Core/Tensor.h"

namespace elman {
struct SamplingOptions {
    double temperature = 1;
    int top_k = 0;
    double top_p = 1;
    double repetition_penalty = 1;
};

// SplitMix64: next() exposes 24 bits, next_double() exposes 53 bits in [0, 1).
// Each request owns its stream; CPU and CUDA consume identical host uniforms.
class SamplingRng {
public:
    explicit SamplingRng(uint64_t seed) : state_(seed) {}
    float next();
    double next_double();
private:
    uint64_t state_;
};

void validate_sampling(const SamplingOptions& options);
// Select from one contiguous vocabulary row. Ties resolve to the lowest ID.
int32_t greedy_token(const Tensor& logits);
// Inference-only [batch, vocabulary] logits and Int32 [batch, history] IDs.
// Double logits retain Double precision; other inputs use Float32 processing.
// The order is repetition, temperature, top-k, then stable top-p.
Tensor sampling_scores(const Tensor& logits, const Tensor& history, const SamplingOptions& options);
// Nonnegative weights and one Float32 uniform in [0, 1) per row. Outputs Int32.
// Low-format weights decode to Float32 before cumulative probabilities are formed.
// Invalid distributions return -1 in the raw kernel and throw at this boundary.
std::vector<int32_t> categorical_sample(const Tensor& weights, const std::vector<float>& uniforms);
// Double weights and uniforms retain full precision through the inverse CDF.
std::vector<int32_t> categorical_sample_double(const Tensor& weights, const std::vector<double>& uniforms);
std::vector<int32_t> sample_tokens(const Tensor& logits, const Tensor& history,
                                 const SamplingOptions& options, SamplingRng& rng);
} // namespace elman
