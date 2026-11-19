#include "kernel.cuh"
#include "Core/AllocationPool.h"
#include "Operators/paged_attention.h"
#include <climits>

namespace elman {
void launch_attention_softmax_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
namespace {
__device__ float read(const uint32_t* data, int index, int type) {
    return elman_cuda_decode(elman_cuda_load(data, index, type), type);
}
__device__ void write(uint32_t* data, int index, float value, int type) {
    elman_cuda_store(data, index, elman_cuda_encode(value, type), type);
}
__global__ void mask_kernel(uint32_t* mask, const int32_t* map, int count, int type, float minimum) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) write(mask, i, map[i] < 0 ? minimum : 0.f, type);
}
__global__ void span_pointers(void** arrays, uint32_t* out, const uint32_t* a, const uint32_t* physical,
                              const int32_t* map, PagedAttentionShape s, int rows, int start, bool scores) {
    const int head = int(blockIdx.x * blockDim.x + threadIdx.x), matrices = s.batch * s.heads;
    if (head >= matrices) return;
    const int batch = head / s.heads, kv_head = (head % s.heads) / (s.heads / s.kv_heads);
    const int code = map[batch * s.keys], slot = code < 0 ? -code - 1 : code;
    const size_t width = s.dtype == int(DType::Float32) ? 4 : 2;
    const size_t right = (size_t(slot) * s.kv_heads + kv_head) * s.dimension;
    const size_t left = scores ? size_t(head * s.queries + start) * s.dimension : size_t(head) * rows * s.keys;
    const size_t destination = scores ? size_t(head) * rows * s.keys : size_t(head * s.queries + start) * s.dimension;
    arrays[head] = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(physical)) + right * width;
    arrays[matrices + head] = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(a)) + left * width;
    arrays[2 * matrices + head] = reinterpret_cast<uint8_t*>(out) + destination * width;
}
__global__ void score_kernel(uint32_t* scores, const uint32_t* q, const uint32_t* k,
                             const int32_t* map, PagedAttentionShape s, int rows, int start) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= s.batch * s.heads * rows * s.keys) return;
    const int row = i / s.keys, head = row / rows, batch = head / s.heads;
    const int kv_head = (head % s.heads) / (s.heads / s.kv_heads);
    const int code = map[batch * s.keys + i % s.keys];
    float dot = 0;
    const int slot = code == INT_MIN ? 0 : code < 0 ? -code - 1 : code;
    const int query = (head * s.queries + start + row % rows) * s.dimension;
    const int key = (slot * s.kv_heads + kv_head) * s.dimension;
    for (int d = 0; d < s.dimension; ++d)
        dot = fmaf(read(q, query + d, s.dtype), code == INT_MIN ? 0.f : read(k, key + d, s.dtype), dot);
    write(scores, i, dot, s.dtype);
}
__global__ void value_kernel(uint32_t* out, const uint32_t* scores, const uint32_t* v,
                             const int32_t* map, PagedAttentionShape s, int rows, int start) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= s.batch * s.heads * rows * s.dimension) return;
    const int row = i / s.dimension, d = i % s.dimension, head = row / rows, batch = head / s.heads;
    const int kv_head = (head % s.heads) / (s.heads / s.kv_heads);
    float sum = 0;
    for (int key = 0; key < s.keys; ++key) {
        const int code = map[batch * s.keys + key];
        const int slot = code < 0 && code != INT_MIN ? -code - 1 : code;
        const float value = code == INT_MIN ? 0.f : read(v, (slot * s.kv_heads + kv_head) * s.dimension + d, s.dtype);
        sum = fmaf(read(scores, row * s.keys + key, s.dtype), value, sum);
    }
    write(out, (head * s.queries + start + row % rows) * s.dimension + d, sum, s.dtype);
}

// One warp reads contiguous channels from each physical page.
template<int Width, class Storage>
__global__ void paged_scores(Storage* scores, const Storage* query, const Storage* keys,
                             const int32_t* map, PagedAttentionShape s, int rows, int start) {
    const unsigned i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i >= unsigned(s.batch * s.heads * rows * s.keys)) return;
    const int lane = threadIdx.x, row = int(i) / s.keys, head = row / rows;
    const int kv_head = (head % s.heads) / (s.heads / s.kv_heads);
    const int code = map[(head / s.heads) * s.keys + int(i % s.keys)];
    const int slot = code >= 0 ? code : -(code + 1);
    const auto q = query + size_t(head * s.queries + start + row % rows) * Width + lane;
    const auto* k = code == INT_MIN ? nullptr : keys + ((size_t(slot) * s.kv_heads + kv_head) * Width + lane);
    float sum = float(q[0]) * (k ? float(k[0]) : 0.f);
    #pragma unroll
    for (int d = 32; d < Width; d += 32) {
        sum = fmaf(float(q[d]), k ? float(k[d]) : 0.f, sum);
    }
    for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (!lane) scores[i] = Storage(sum);
}

template<int Width, class Storage>
__global__ void paged_values(Storage* out, const Storage* scores, const Storage* values,
                             const int32_t* map, PagedAttentionShape s, int rows, int start) {
    constexpr int warps = 8;
    constexpr int channels_per_lane = Width / 32;
    __shared__ float partial[warps][Width];
    const int row = int(blockIdx.x), head = row / rows, lane = threadIdx.x, warp = threadIdx.y;
    const int kv_head = (head % s.heads) / (s.heads / s.kv_heads);
    const int32_t* slots = map + (head / s.heads) * s.keys;
    const Storage* weights = scores + row * s.keys;
    float sum[channels_per_lane]{};
    for (unsigned key = warp; key < unsigned(s.keys); key += warps) {
        const int code = slots[key];
        const int slot = code < 0 ? -(code + 1) : code;
        const auto* v = code == INT_MIN ? nullptr : values + ((size_t(slot) * s.kv_heads + kv_head) * Width + lane);
        const float weight = float(weights[key]);
        #pragma unroll
        for (int channel = 0; channel < channels_per_lane; ++channel) {
            sum[channel] = fmaf(weight, v ? float(v[channel * 32]) : 0.f, sum[channel]);
        }
    }
    #pragma unroll
    for (int channel = 0; channel < channels_per_lane; ++channel) {
        partial[warp][lane + channel * 32] = sum[channel];
    }
    __syncthreads();
    if (warp) return;
    auto destination = out + ((head * s.queries + start + row % rows) * Width + lane);
    #pragma unroll
    for (int channel = 0; channel < channels_per_lane; ++channel) {
        const int d = lane + channel * 32;
        for (int w = 1; w < warps; ++w) sum[channel] += partial[w][d];
        destination[channel * 32] = Storage(sum[channel]);
    }
}

template<int Width, class Storage>
void launch_warp_paged_tile(uint32_t* out, uint32_t* scores, const uint32_t* q, const uint32_t* k,
                           const uint32_t* v, const uint32_t* mask, const int32_t* map,
                           const PagedAttentionShape& shape, int rows, int start, const KernelParams& p) {
    const int query_rows = shape.batch * shape.heads * rows;
    constexpr int warps = 8;
    auto stream = current_cuda_stream();
    auto* typed_scores = reinterpret_cast<Storage*>(scores);
    paged_scores<Width><<<(size_t(query_rows) * shape.keys + warps - 1) / warps, dim3(32,warps), 0, stream>>>(
        typed_scores, reinterpret_cast<const Storage*>(q), reinterpret_cast<const Storage*>(k), map, shape, rows, start);
    launch_attention_softmax_cuda(scores, scores, mask, nullptr, shape.probabilities(p, rows, start));
    paged_values<Width><<<query_rows, dim3(32,warps), 0, stream>>>(
        reinterpret_cast<Storage*>(out), typed_scores, reinterpret_cast<const Storage*>(v), map, shape, rows, start);
}
}
void launch_paged_attention_cuda(uint32_t* out, const uint32_t* q, const uint32_t* k,
                                const uint32_t* v, const uint32_t* encoded, const KernelParams& p) {
    if (!p.output_count()) return;
    const PagedAttentionShape s(p);
    const size_t bytes = dtype_storage_bytes(DType(s.dtype), 1);
    if (!s.keys) {
        check_cuda_execution(cudaMemsetAsync(out, 0, size_t(p.output_count()) * bytes, current_cuda_stream()), "empty paged attention");
        return;
    }
    auto score_buffer = backend_workspace(Backend::CUDA, size_t(s.batch) * s.heads * s.tile * s.keys * bytes);
    auto mask_buffer = backend_workspace(Backend::CUDA, size_t(s.batch) * s.keys * bytes);
    auto* scores = static_cast<uint32_t*>(score_buffer->raw_pointer());
    auto* mask = static_cast<uint32_t*>(mask_buffer->raw_pointer());
    const auto* map = reinterpret_cast<const int32_t*>(encoded);
    const float minimum = s.dtype == int(DType::Float16) ? -65504.f :
        s.dtype == int(DType::BFloat16) ? -3.3895313892515355e38f : -3.402823466e38f;
    const int mask_count = s.batch * s.keys;
    mask_kernel<<<(size_t(mask_count) + 255) / 256, 256, 0, current_cuda_stream()>>>(mask, map, mask_count, s.dtype, minimum);
    if (p.attribute(5)) {
        const int matrices = s.batch * s.heads;
        auto pointer_buffer = backend_workspace(Backend::CUDA, size_t(matrices) * 3 * sizeof(void*));
        auto arrays = static_cast<void**>(pointer_buffer->raw_pointer());
        for (int next = 0; next < s.queries; next += std::min(s.tile, s.queries - next)) {
            const int rows = s.tile, start = std::min(next, s.queries - rows);
            AttentionGemm qk{DType(s.dtype), rows, s.keys, s.dimension, matrices, s.heads / s.kv_heads,
                            s.queries * s.dimension, 0, rows * s.keys, true};
            AttentionGemm pv{DType(s.dtype), rows, s.dimension, s.keys, matrices, s.heads / s.kv_heads,
                            rows * s.keys, 0, s.queries * s.dimension, false};
            qk.b_leading = pv.b_leading = s.kv_heads * s.dimension;
            span_pointers<<<(matrices + 255u) / 256u, 256, 0, current_cuda_stream()>>>(arrays, scores, q, k, map, s, rows, start, true);
            attention_gemm_cuda_pointers(arrays, qk);
            launch_attention_softmax_cuda(scores, scores, mask, nullptr, s.probabilities(p, rows, start));
            span_pointers<<<(matrices + 255u) / 256u, 256, 0, current_cuda_stream()>>>(arrays, out, scores, v, map, s, rows, start, false);
            attention_gemm_cuda_pointers(arrays, pv);
        }
        return;
    }
    for (int start = 0; start < s.queries; start += s.tile) {
        const int rows = std::min(s.tile, s.queries - start);
        const int scores_count = s.batch * s.heads * rows * s.keys, output_count = s.batch * s.heads * rows * s.dimension;
        if (s.keys >= 64) {
            if (s.dtype == int(DType::Float32) && s.dimension == 64) {
                launch_warp_paged_tile<64,float>(out,scores,q,k,v,mask,map,s,rows,start,p);
                continue;
            }
            if (s.queries == 1 && s.dimension == 128 && s.keys >= 256) {
                if (s.dtype == int(DType::BFloat16)) {
                    launch_warp_paged_tile<128,__nv_bfloat16>(out,scores,q,k,v,mask,map,s,rows,start,p);
                    continue;
                }
                if (s.dtype == int(DType::Float16)) {
                    launch_warp_paged_tile<128,__half>(out,scores,q,k,v,mask,map,s,rows,start,p);
                    continue;
                }
            }
        }
        score_kernel<<<(size_t(scores_count) + 255) / 256, 256, 0, current_cuda_stream()>>>(scores, q, k, map, s, rows, start);
        launch_attention_softmax_cuda(scores, scores, mask, nullptr, s.probabilities(p, rows, start));
        value_kernel<<<(size_t(output_count) + 255) / 256, 256, 0, current_cuda_stream()>>>(out, scores, v, map, s, rows, start);
    }
}
} // namespace elman
