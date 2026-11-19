#include "kernel.cuh"
#include "Core/AllocationPool.h"
#include "Operators/attention_params.h"

#include <cstring>

namespace elman {
namespace {
constexpr int lanes = 32;
constexpr int query_warps = 4;
constexpr int rows_per_warp = 4;
constexpr int query_rows = query_warps * rows_per_warp;
constexpr int keys_per_tile = 32;

struct OnlineAttention {
    int query_count, key_count, head_count, groups;
    int mask_stride[4];
    int causal_offset;
    bool causal;
    float scale;
};

template<int head_width, typename Storage, bool masked>
__global__ void online_attention(Storage* out, const Storage* q, const Storage* k,
                                 const Storage* v, const float* mask, OnlineAttention s) {
    __shared__ float queries[query_rows][head_width];
    __shared__ float keys[head_width][keys_per_tile + 1];
    __shared__ float values[keys_per_tile][head_width];
    __shared__ float weights[query_rows][keys_per_tile];
    const int lane = threadIdx.x % lanes, warp = threadIdx.x / lanes;
    const int first = int(blockIdx.x) * query_rows;
    const int head = blockIdx.y, kv_head = head / s.groups;
    const size_t query_base = (size_t(head) * s.query_count + first) * head_width;
    for (int i = threadIdx.x; i < query_rows * head_width; i += blockDim.x) {
        const int row = i / head_width;
        queries[row][i % head_width] = first + row < s.query_count ? float(q[query_base + i]) : 0.f;
    }
    float maximum[rows_per_warp], total[rows_per_warp]{};
    float output[head_width / lanes][rows_per_warp]{};
    #pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) maximum[r] = -EL_INF;
    for (int tile = 0; tile < s.key_count; tile += keys_per_tile) {
        const int count = min(keys_per_tile, s.key_count - tile);
        const size_t source = (size_t(kv_head) * s.key_count + tile) * head_width;
        for (int i = threadIdx.x; i < keys_per_tile * head_width; i += blockDim.x) {
            const int key = i / head_width, d = i % head_width;
            keys[d][key] = key < count ? float(k[source+i]) : 0.f;
            values[key][d] = key < count ? float(v[source+i]) : 0.f;
        }
        __syncthreads();
        // Each lane owns one key. The warp reuses its keys across several
        // query rows and reduces the softmax only once per key tile.
        float scores[rows_per_warp]{};
        #pragma unroll
        for (int d = 0; d < head_width; ++d) {
            const float key = keys[d][lane];
            #pragma unroll
            for (int r = 0; r < rows_per_warp; ++r)
                scores[r] = fmaf(queries[warp * rows_per_warp + r][d], key, scores[r]);
        }
        #pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int row = warp * rows_per_warp + r, query = first + row;
            const int key = tile + lane;
            float score = -EL_INF;
            if (query < s.query_count && lane < count) {
                float offset = 0;
                if constexpr (masked) {
                    const size_t mask_base = size_t(head / s.head_count) * s.mask_stride[0] +
                                             size_t(head % s.head_count) * s.mask_stride[1];
                    offset = s.causal && int64_t(key) > int64_t(query) + s.causal_offset ?
                        -3.402823466e38f : mask[mask_base + size_t(query)*s.mask_stride[2] + size_t(key)*s.mask_stride[3]];
                }
                score = __fadd_rn(__fmul_rn(scores[r],s.scale),offset);
            }
            float next = fmaxf(maximum[r],score);
            #pragma unroll
            for (int step = lanes/2; step; step /= 2)
                next = fmaxf(next,__shfl_xor_sync(0xffffffff,next,step));
            const float carry = next == maximum[r] ? 1.f : __expf(maximum[r]-next);
            const float weight = score == -EL_INF ? 0.f : __expf(score-next);
            weights[row][lane] = weight;
            float sum = weight;
            #pragma unroll
            for (int step = lanes/2; step; step /= 2) sum += __shfl_xor_sync(0xffffffff,sum,step);
            total[r] = fmaf(carry,total[r],sum);
            maximum[r] = next;
            #pragma unroll
            for (int col = 0; col < head_width / lanes; ++col) output[col][r] *= carry;
        }
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < keys_per_tile; ++j) {
            #pragma unroll
            for (int col = 0; col < head_width / lanes; ++col) {
                const float value = values[j][lane + col * lanes];
                #pragma unroll
                for (int r = 0; r < rows_per_warp; ++r)
                    output[col][r] = fmaf(weights[warp * rows_per_warp + r][j],value,output[col][r]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
        const int row = warp * rows_per_warp + r;
        if (first + row < s.query_count) {
            const size_t destination = query_base + row * head_width + lane;
            #pragma unroll
            for (int col = 0; col < head_width / lanes; ++col)
                out[destination + col * lanes] = Storage(output[col][r] / total[r]);
        }
    }
}

bool try_online_attention(uint32_t* out, const uint32_t* q, const uint32_t* k,
                          const uint32_t* v, const uint32_t* mask,
                          const TiledAttentionShape& s, const KernelParams& p) {
    if (s.dtype != DType::Float32 || s.dimension != 64 || s.values != 64 ||
        s.queries < 32 || s.keys < keys_per_tile || s.keys > 2048 || s.batch * s.heads > 65535 ||
        p.attribute(operator_attr::tiled_attention::kTiling) != 0)
        return false;
    OnlineAttention options{};
    options.query_count = s.queries;
    options.key_count = s.keys;
    options.head_count = s.heads;
    options.groups = s.groups;
    options.causal = p.attribute(operator_attr::tiled_attention::kCausal) != 0;
    options.causal_offset = p.attribute(operator_attr::tiled_attention::kCausalOffset);
    const int scale_bits = p.attribute(operator_attr::tiled_attention::kDivisor);
    float divisor;
    std::memcpy(&divisor, &scale_bits, sizeof(divisor));
    options.scale = 1.f / divisor;
    const int rank = p.input_rank(KernelInput::D);
    for (int axis = 0; axis < rank; ++axis)
        if (p.input_shape(KernelInput::D, axis) != 1)
            options.mask_stride[4 - rank + axis] = p.input_stride(KernelInput::D, axis);
    const dim3 grid((unsigned(s.queries) + query_rows - 1) / query_rows, s.batch * s.heads);
    online_attention<64, float, true><<<grid, lanes * query_warps, 0, current_cuda_stream()>>>(reinterpret_cast<float*>(out),
        reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v),
        reinterpret_cast<const float*>(mask), options);
    return true;
}

AttentionGemm merge_grouped_query_rows(AttentionGemm shape) {
    // Full query matrices are adjacent within each group that shares one KV head.
    shape.rows *= shape.groups;
    shape.batches /= shape.groups;
    shape.a_stride *= shape.groups;
    shape.output_stride *= shape.groups;
    shape.groups = 1;
    return shape;
}

template<typename Storage>
void flux_attention(uint32_t* out, const uint32_t* q, const uint32_t* k, const uint32_t* v, const KernelParams& p) {
    OnlineAttention s{};
    s.query_count = p.output_shape(2);
    s.key_count = p.input_shape(KernelInput::B, 2);
    s.head_count = p.output_shape(1);
    s.groups = 1;
    s.scale = 1.f / sqrtf(128.f);
    const int heads = p.output_shape(0) * s.head_count;
    for (int start = 0; start < heads; start += 65535) {
        const size_t query_offset = size_t(start) * s.query_count * 128;
        const size_t key_offset = size_t(start) * s.key_count * 128;
        const dim3 grid((unsigned(s.query_count) + query_rows - 1) / query_rows, std::min(heads - start, 65535));
        online_attention<128, Storage, false><<<grid, lanes * query_warps, 0, current_cuda_stream()>>>(
            reinterpret_cast<Storage*>(out) + query_offset, reinterpret_cast<const Storage*>(q) + query_offset,
            reinterpret_cast<const Storage*>(k) + key_offset, reinterpret_cast<const Storage*>(v) + key_offset, nullptr, s);
    }
}
} // namespace

void launch_flux_attention_cuda(uint32_t* out, const uint32_t* q, const uint32_t* k, const uint32_t* v, const KernelParams& p) {
    if (!p.output_count()) return;
    if (p.output_dtype() == int(DType::BFloat16)) flux_attention<__nv_bfloat16>(out, q, k, v, p);
    else if (p.output_dtype() == int(DType::Float16)) flux_attention<__half>(out, q, k, v, p);
    else flux_attention<float>(out, q, k, v, p);
}

void launch_attention_softmax_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
void launch_tiled_attention_cuda(uint32_t* out, const uint32_t* q, const uint32_t* k,
                                 const uint32_t* v, const uint32_t* mask, const KernelParams& p) {
    if (!p.output_count() || !p.input_shape(KernelInput::B, 2)) return;
    TiledAttentionShape s(p);
    if (try_online_attention(out, q, k, v, mask, s, p)) return;
    size_t bytes = dtype_storage_bytes(s.dtype, 1);
    if (p.attribute(operator_attr::tiled_attention::kTiling) == 1) {
        auto workspace = backend_workspace(Backend::CUDA, checked_memory_mul(size_t(s.queries) * s.keys, bytes));
        auto scores = static_cast<uint32_t*>(workspace->raw_pointer());
        auto qk = s.scores(s.queries), pv = s.attended(s.queries);
        qk.batches = pv.batches = qk.groups = pv.groups = 1;
        auto probabilities = s.probabilities(p, s.queries, 0, true);
        for (int head = 0; head < s.batch * s.heads; ++head) {
            size_t kv_head = size_t(head / s.groups);
            attention_gemm_cuda(scores, attention_offset(q, size_t(head) * s.queries * s.dimension, bytes),
                attention_offset(k, kv_head * s.keys * s.dimension, bytes), nullptr, qk);
            launch_attention_softmax_cuda(scores, scores, attention_offset(mask, s.head_mask_offset(p, head), bytes), nullptr, probabilities);
            attention_gemm_cuda(attention_offset(out, size_t(head) * s.queries * s.values, bytes), scores,
                attention_offset(v, kv_head * s.keys * s.values, bytes), nullptr, pv);
        }
        return;
    }
    auto workspace = backend_workspace(Backend::CUDA, checked_memory_mul(size_t(s.batch) * s.heads * s.tile * s.keys, bytes));
    auto scores = static_cast<uint32_t*>(workspace->raw_pointer());
    const bool merge_heads = s.groups > 1 && s.tile == s.queries;
    const auto scores_shape = s.scores(s.tile);
    const auto attended_shape = s.attended(s.tile);
    const auto qk = merge_heads ? merge_grouped_query_rows(scores_shape) : scores_shape;
    const auto pv = merge_heads ? merge_grouped_query_rows(attended_shape) : attended_shape;
    BufferPtr pointers;
    if (qk.groups > 1) pointers = backend_workspace(Backend::CUDA, checked_memory_mul(size_t(s.batch) * s.heads, 3 * sizeof(void*)));
    void* arrays = pointers ? pointers->raw_pointer() : nullptr;
    for (int next = 0; next < s.queries;) {
        // A short tail can select GEMV arithmetic instead of the full tile's
        // GEMM reduction. Overlap the previous rows to retain the tile shape
        // and rounding without allocating a padded query or larger workspace.
        int rows = s.tile, start = std::min(next, s.queries - rows);
        attention_gemm_cuda(scores, attention_offset(q, size_t(start) * s.dimension, bytes), k, arrays, qk);
        // Contiguous softmax loads/reduces a complete row before its final
        // stores, so scores and probabilities can share the same tile.
        launch_attention_softmax_cuda(scores, scores, attention_offset(mask, s.mask_offset(p, start), bytes), nullptr, s.probabilities(p, rows, start));
        attention_gemm_cuda(attention_offset(out, size_t(start) * s.values, bytes), scores, v, arrays, pv);
        next += std::min(s.tile, s.queries - next);
    }
}
} // namespace elman
