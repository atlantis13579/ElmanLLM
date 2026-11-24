#pragma once

#include "Inference/LLM/Generation.h"
#include "Inference/LLM/Qwen2Graph.h"
#include <algorithm>

namespace elman::detail {
inline void add_graph_memory(DecoderMemoryPlan& plan, const DecoderConfig& config, DType dtype,
                      const GenerationOptions& options, size_t weight_workspace) {
    if (!options.cuda_graph) return;
    // Single-device eager and recorded forwards request the same fixed-size
    // quantization workspace. Its lease is released at forward exit and reused
    // by the graph, with cross-stream dependencies retained by the pool.
    const size_t shared_workspace = std::min({plan.additional_bytes, plan.weight_workspace_bytes, weight_workspace});
    plan.additional_bytes = checked_memory_add(plan.additional_bytes,
        Qwen2Graph::memory_bound(config, dtype, options.max_context, weight_workspace) - shared_workspace);
    plan.host_additional_bytes = checked_memory_add(plan.host_additional_bytes, 16 * 1024 * 1024);
}
}
