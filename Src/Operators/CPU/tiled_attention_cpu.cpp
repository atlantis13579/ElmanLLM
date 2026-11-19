#include "kernel.h"
#include "Core/AllocationPool.h"
#include "Operators/attention_params.h"
#include "row_reduction.h"

namespace elman {
void launch_attention_softmax_cpu(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
namespace {
void attention_gemm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const AttentionGemm& s) {
    if (try_attention_gemm_cpu(out, a, b, s)) return;
    for (int batch = 0; batch < s.batches; ++batch)
        for (int row = 0; row < s.rows; ++row)
            for (int column = 0; column < s.columns; ++column) {
                float value = 0;
                auto left = [&](int k) { return cpu_read(a, batch * s.a_stride + row * s.shared + k, int(s.dtype)); };
                auto right = [&](int k) { return cpu_read(b, (batch / s.groups) * s.b_stride +
                    (s.transpose_b ? column * s.shared + k : k * s.columns + column), int(s.dtype)); };
                // Retain the public batched matrix primitive's fallback order.
                if (s.dtype == DType::Float32 && (s.transpose_b || s.columns == 1) &&
                    int64_t(s.rows) * s.columns * s.shared >= 400)
                    value = contiguous_float_dot(s.shared, left, right);
                else for (int k = 0; k < s.shared; ++k) value += left(k) * right(k);
                cpu_write(out, batch * s.output_stride + row * s.columns + column, value, int(s.dtype));
            }
}
}
void launch_tiled_attention_cpu(uint32_t* out, const uint32_t* q, const uint32_t* k,
                                const uint32_t* v, const uint32_t* mask, const KernelParams& p) {
    if (!p.output_count() || !p.input_shape(KernelInput::B, 2)) return;
    TiledAttentionShape s(p);
    size_t bytes = dtype_storage_bytes(s.dtype, 1);
    if (p.attribute(operator_attr::tiled_attention::kTiling) == 1) {
        auto workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(s.queries) * s.keys, bytes));
        auto scores = static_cast<uint32_t*>(workspace->raw_pointer());
        auto qk = s.scores(s.queries), pv = s.attended(s.queries);
        qk.batches = pv.batches = qk.groups = pv.groups = 1;
        qk.serial_cpu = pv.serial_cpu = s.batch * s.heads > 1;
        auto probabilities = s.probabilities(p, s.queries, 0, true);
        for (int head = 0; head < s.batch * s.heads; ++head) {
            size_t kv_head = size_t(head / s.groups);
            attention_gemm_cpu(scores, attention_offset(q, size_t(head) * s.queries * s.dimension, bytes),
                attention_offset(k, kv_head * s.keys * s.dimension, bytes), qk);
            launch_attention_softmax_cpu(scores, scores, attention_offset(mask, s.head_mask_offset(p, head), bytes), nullptr, probabilities);
            attention_gemm_cpu(attention_offset(out, size_t(head) * s.queries * s.values, bytes), scores,
                attention_offset(v, kv_head * s.keys * s.values, bytes), pv);
        }
        return;
    }
    auto workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(s.batch) * s.heads * s.tile * s.keys, bytes));
    auto scores = static_cast<uint32_t*>(workspace->raw_pointer());
    for (int start = 0; start < s.queries;) {
        int rows = std::min(s.tile, s.queries - start);
        attention_gemm_cpu(scores, attention_offset(q, size_t(start) * s.dimension, bytes), k, s.scores(rows));
        // Each complete score row is reduced before it is overwritten. Keep
        // the original mask batch/head strides and shift only its query base.
        launch_attention_softmax_cpu(scores, scores, attention_offset(mask, s.mask_offset(p, start), bytes), nullptr, s.probabilities(p, rows, start));
        attention_gemm_cpu(attention_offset(out, size_t(start) * s.values, bytes), scores, v, s.attended(rows));
        start += rows;
    }
}
} // namespace elman
