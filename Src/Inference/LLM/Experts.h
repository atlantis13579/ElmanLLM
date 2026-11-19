#pragma once

#include "Core/Tensor.h"
#include <functional>

namespace elman {
enum class ExpertScore { Softmax, Sigmoid };
enum class ExpertGroupScore { Maximum, TopTwo };

struct ExpertRoutingOptions {
    int experts = 0, top_k = 0, groups = 1, selected_groups = 1;
    ExpertScore score = ExpertScore::Softmax;
    ExpertGroupScore group_score = ExpertGroupScore::Maximum;
    bool normalize = false;
    bool scale_normalized = true;
    bool zero_masked_groups = false;
    float scale = 1;
    float normalization_epsilon = 1e-20f;
    void validate() const;
};

struct ExpertRoutes {
    Tensor indices; // Int32 [tokens, top_k], stable lower-index tie breaking.
    Tensor weights; // Float32 [tokens, top_k], before the expert output cast.
    int experts = 0;
};

// Routing scores, group selection and weights execute on the operand backend.
// Correction bias affects selection only, never the returned mixture weights.
ExpertRoutes route_experts(const Tensor& logits, const ExpertRoutingOptions& options,
                          const Tensor& correction_bias = {});

// Sparse inference: invoke only experts receiving tokens. Host metadata contains
// route IDs and row lists, never hidden activations or expert calculations.
// The callback returns [selected_tokens, hidden] on the input backend/dtype.
Tensor evaluate_experts(const Tensor& input, const ExpertRoutes& routes,
                        const std::function<Tensor(int, const Tensor&)>& expert);
} // namespace elman
