#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Backends/CpuMath.h"
#include "Backends/CpuParallel.h"
#include "Core/AllocationPool.h"
#include "Kernals/paged_attention.h"
#include "row_reduction.h"
#include <array>
#include <climits>

namespace elman {
void launch_attention_softmax_cpu(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
namespace {
const uint32_t* offset(const uint32_t* pointer, size_t elements, size_t bytes) {
    return reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(pointer) + elements * bytes);
}
uint32_t* offset(uint32_t* pointer, size_t elements, size_t bytes) {
    return const_cast<uint32_t*>(offset(static_cast<const uint32_t*>(pointer), elements, bytes));
}
void span_product(uint32_t* out, const uint32_t* a, const uint32_t* b, const AttentionGemm& s) {
    if (try_attention_gemm_cpu(out, a, b, s)) return;
    const int type = int(s.dtype);
    for (int row = 0; row < s.rows; ++row) for (int column = 0; column < s.columns; ++column) {
        LocalFunction left{[&](int k) { return cpu_read(a, row * s.shared + k, type); }};
        LocalFunction right{[&](int k) { return cpu_read(b, s.transpose_b ? column * s.b_leading + k : k * s.b_leading + column, type); }};
        float sum = 0;
        if (s.dtype == DType::Float32 && (s.transpose_b || s.columns == 1) && int64_t(s.rows) * s.columns * s.shared >= 400)
            sum = contiguous_float_dot(s.shared, left, right);
        else for (int k = 0; k < s.shared; ++k) sum += left(k) * right(k);
        cpu_write(out, row * s.columns + column, sum, type);
    }
}
template<class Function> void parallel_rows(int count, Function function) {
    parallel_for_cpu(count, 1, [&](int64_t begin, int64_t end) {
        for (int row = static_cast<int>(begin); row < end; ++row) function(row);
    });
}

constexpr int decode_width = 128;
template<DType Type>
void accumulate_decode_values(uint32_t* out, const uint32_t* scores, const uint32_t* values,
                              const int32_t* map, int keys, int kv_heads, int kv_head) {
    constexpr int dtype = int(Type);
    std::array<float, decode_width> sums{};
    for (int key = 0; key < keys; ++key) {
        const int code = map[key];
        const float weight = cpu_read(scores, key, dtype);
        if (code == INT_MIN) {
            for (float& sum : sums) sum = std::fma(weight, 0.f, sum);
            continue;
        }
        const int slot = code < 0 ? -code - 1 : code;
        const int base = (slot * kv_heads + kv_head) * decode_width;
        for (int d = 0; d < decode_width; ++d)
            sums[d] = std::fma(weight, cpu_read(values, base + d, dtype), sums[d]);
    }
    for (int d = 0; d < decode_width; ++d) cpu_write(out, d, sums[d], dtype);
}
}
void launch_paged_attention_cpu(uint32_t* out, const uint32_t* q, const uint32_t* k,
                               const uint32_t* v, const uint32_t* encoded, const KernelParams& p) {
    if (!p.output_count()) return;
    const PagedAttentionShape s(p);
    if (!s.keys) {
        for (int i = 0; i < p.output_count(); ++i) cpu_write(out, i, 0, s.dtype);
        return;
    }
    const int32_t* map = reinterpret_cast<const int32_t*>(encoded);
    const size_t bytes = dtype_storage_bytes(DType(s.dtype), 1);
    BufferPtr score_buffer = backend_workspace(Backend::CPU, size_t(s.tile) * s.keys * bytes);
    BufferPtr mask_buffer = backend_workspace(Backend::CPU, size_t(s.batch) * s.keys * bytes);
    uint32_t* scores = static_cast<uint32_t*>(score_buffer->raw_pointer());
    uint32_t* mask = static_cast<uint32_t*>(mask_buffer->raw_pointer());
    const float minimum = s.dtype == int(DType::Float16) ? -65504.f :
        s.dtype == int(DType::BFloat16) ? -3.3895313892515355e38f : -3.402823466e38f;
    for (int i = 0; i < s.batch * s.keys; ++i) cpu_write(mask, i, map[i] < 0 ? minimum : 0.f, s.dtype);
    for (int head = 0; head < s.batch * s.heads; ++head) for (int start = 0; start < s.queries; start += s.tile) {
        const int rows = std::min(s.tile, s.queries - start);
        const int batch = head / s.heads, kv_head = (head % s.heads) / (s.heads / s.kv_heads);
        if (p.attribute(5)) {
            const int code = map[batch * s.keys], slot = code < 0 ? -code - 1 : code;
            const int leading = s.kv_heads * s.dimension;
            const size_t physical = size_t(slot) * leading + kv_head * s.dimension;
            AttentionGemm qk{DType(s.dtype), rows, s.keys, s.dimension, 1, 1, 0, 0, 0, true};
            AttentionGemm pv{DType(s.dtype), rows, s.dimension, s.keys, 1, 1, 0, 0, 0, false};
            qk.b_leading = pv.b_leading = leading;
            qk.serial_cpu = pv.serial_cpu = s.batch * s.heads > 1;
            span_product(scores, offset(q, size_t(head * s.queries + start) * s.dimension, bytes), offset(k, physical, bytes), qk);
            const uint32_t* head_mask = offset(mask, size_t(batch) * s.keys, bytes);
            launch_attention_softmax_cpu(scores, scores, head_mask, nullptr, s.probabilities(p, rows, start, true));
            span_product(offset(out, size_t(head * s.queries + start) * s.dimension, bytes), scores, offset(v, physical, bytes), pv);
            continue;
        }
        parallel_rows(rows, [&](int row) {
            const int query = (head * s.queries + start + row) * s.dimension;
            for (int key = 0; key < s.keys; ++key) {
                const int code = map[batch * s.keys + key];
                float dot = 0;
                const int slot = code == INT_MIN ? 0 : code < 0 ? -code - 1 : code;
                const int offset = (slot * s.kv_heads + kv_head) * s.dimension;
                for (int d = 0; d < s.dimension; ++d)
                    dot = std::fma(cpu_read(q, query + d, s.dtype), code == INT_MIN ? 0.f : cpu_read(k, offset + d, s.dtype), dot);
                cpu_write(scores, row * s.keys + key, dot, s.dtype);
            }
        });
        const uint32_t* head_mask = offset(mask, size_t(batch) * s.keys, bytes);
        launch_attention_softmax_cpu(scores, scores, head_mask, nullptr, s.probabilities(p, rows, start, true));
        if (s.queries == 1 && s.dimension == decode_width && s.keys >= 256) {
            uint32_t* output = offset(out, size_t(head) * decode_width, bytes);
            const int32_t* head_map = map + batch * s.keys;
            if (s.dtype == int(DType::Float16)) {
                accumulate_decode_values<DType::Float16>(output, scores, v, head_map, s.keys, s.kv_heads, kv_head);
                continue;
            }
            if (s.dtype == int(DType::BFloat16)) {
                accumulate_decode_values<DType::BFloat16>(output, scores, v, head_map, s.keys, s.kv_heads, kv_head);
                continue;
            }
        }
        parallel_rows(rows, [&](int row) {
            const int output = (head * s.queries + start + row) * s.dimension;
            for (int d = 0; d < s.dimension; ++d) {
                float sum = 0;
                for (int key = 0; key < s.keys; ++key) {
                    const int code = map[batch * s.keys + key];
                    const int slot = code < 0 && code != INT_MIN ? -code - 1 : code;
                    const float value = code == INT_MIN ? 0.f : cpu_read(v, (slot * s.kv_heads + kv_head) * s.dimension + d, s.dtype);
                    sum = std::fma(cpu_read(scores, row * s.keys + key, s.dtype), value, sum);
                }
                cpu_write(out, output + d, sum, s.dtype);
            }
        });
    }
}
} // namespace elman
