#pragma once

#include "Core/Memory.h"
#include "Core/AllocationPool.h"
#include "Models/LLM/Qwen2.h"
#include <memory>

namespace elman {
struct MemoryBudget {
    // Absolute tracked bytes, including already resident tensors. Zero is automatic.
    size_t limit_bytes = 0;
    size_t reserve_bytes = 256 * 1024 * 1024;
};
struct MemoryReport {
    size_t budget_bytes = 0, estimated_peak_bytes = 0, measured_peak_bytes = 0;
    size_t host_budget_bytes = 0, host_estimated_peak_bytes = 0, host_measured_peak_bytes = 0;
};
struct DecoderMemoryPlan {
    size_t parameter_bytes = 0, kv_cache_bytes = 0, weight_workspace_bytes = 0;
    size_t prefill_bytes = 0, decode_bytes = 0, selection_bytes = 0;
    size_t additional_bytes = 0, host_additional_bytes = 0;
};

// Conservative bounds for eager or tiled execution, without hidden-state
// capture or user callbacks. CPU additional_bytes includes host workspaces.
DecoderMemoryPlan plan_decoder_generation(
    const DecoderConfig& config, Backend backend, DType dtype, size_t prompt_tokens, size_t output_tokens,
    bool sampled, bool tiled_attention = false, bool packed_projections = true, bool paged_cache = false,
    int prefill_chunk_size = 0, bool full_logits = true, size_t weight_workspace_bytes = 0);
DecoderMemoryPlan plan_decoder_weight_load(const std::string& directory, Backend backend, DType dtype,
                                           bool packed_projections = true);
struct DeviceLoadPlan { int ordinal; DecoderMemoryPlan memory; };
DeviceLoadPlan plan_qwen_tensor_parallel_load(const std::string& directory, Backend backend, DType dtype,
    bool packed_projections, int rank, int ranks);
std::vector<DeviceLoadPlan> plan_decoder_load(const std::string& directory, Backend backend, DType dtype,
    bool packed_projections, const std::vector<int>& devices);
// One grouped paged forward, without captures or callback allocations. Model,
// physical pool storage and resident request state are counted by admission.
// Sampling executes one row at a time while the batch logits remain alive.
// Include a private copy of each shared partial tail. Physical replacement
// pages already belong to the resident pool; only copy scratch is additional.
struct QwenBatchMemoryPlan {
    size_t forward_bytes = 0, selection_bytes = 0;
    size_t additional_bytes = 0, host_additional_bytes = 0;
};
QwenBatchMemoryPlan plan_qwen_batch(const DecoderConfig& config, Backend backend, DType dtype,
                                    size_t batch, size_t query, size_t total, Qwen2Logits logits,
                                    bool sampled, bool tiled_attention = true, int cache_block_size = 16, size_t weight_workspace_bytes = 0);
size_t qwen_host_workspace(const DecoderConfig& config, size_t batch, size_t sequence, size_t total,
                           bool full_logits = true, bool tiled_attention = false, bool head_tiling = false);

// Admission is a snapshot, followed by runtime payload/workspace ceilings. It
// cannot guarantee allocation success if other processes consume available RAM.
class MemoryAdmission {
public:
    MemoryAdmission(Backend backend, const MemoryBudget& budget, size_t additional_bytes, size_t host_additional_bytes = 0);
    MemoryReport report() const;
private:
    Backend backend_;
    MemoryReport report_;
    std::unique_ptr<MemoryLimit> limit_, host_limit_;
};

struct DeviceMemoryReport { int ordinal; MemoryReport memory; };
// The same explicit ceiling applies separately to every device. Forward scratch
// is conservatively reserved in full on each stage; resident weights are local.
class DecoderMemoryAdmission {
public:
    DecoderMemoryAdmission(Backend backend, const MemoryBudget& budget,
        const std::vector<DeviceLoadPlan>& plans, bool loading);
    DecoderMemoryAdmission(const DecoderModel& model, const MemoryBudget& budget, const DecoderMemoryPlan& forward);
    MemoryReport report() const;
    std::vector<DeviceMemoryReport> device_reports() const;
    void enable_pools(bool reuse);
    AllocationPoolStats allocation_stats() const;
private:
    Backend backend_;
    std::vector<int> ordinals_;
    std::vector<std::unique_ptr<MemoryAdmission>> admissions_;
    std::vector<std::unique_ptr<AllocationPool>> pools_;
};
} // namespace elman
