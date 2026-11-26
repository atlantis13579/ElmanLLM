#include "Kernals/float64_attention_math.h"
#include "Core/LocalFunction.h"
#include "Backends/CudaStream.h"
#include "Core/AllocationPool.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace elman {
__global__ void float64_attention_softmax_kernel(double* output, const double* input, const double* mask,
        int rows, int length, int inner, KernelParams p) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < unsigned(rows)) float64_attention_softmax_row(output, input, mask, int(row), length, inner, p);
}
__global__ void float64_attention_scores_kernel(double* output, const double* query, const double* key,
        int count, int first_head, int start, int rows, Float64AttentionShape s) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < unsigned(count)) output[i] = float64_attention_score(query, key,
        first_head + int(i) / (rows * s.keys), start + int(i) / s.keys % rows, int(i) % s.keys, s);
}
__global__ void float64_attention_probabilities_kernel(double* scores, const double* mask,
        int count, int first_head, int start, int rows, Float64AttentionShape s, KernelParams p) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < unsigned(count)) float64_attention_probabilities(scores + i * s.keys, mask,
        first_head + int(i) / rows, start + int(i) % rows, s, p);
}
__global__ void float64_attention_values_kernel(double* output, const double* scores, const double* values,
        int count, int first_head, int start, int rows, Float64AttentionShape s) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= unsigned(count)) return;
    const int head = first_head + int(i) / (rows * s.values), row = int(i) / s.values % rows, column = int(i) % s.values;
    output[(head * s.queries + start + row) * s.values + column] =
        float64_attention_value(scores + (int(i) / s.values) * s.keys, values, head, column, s);
}
__global__ void float64_attention_zero_kernel(double* output, int count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < unsigned(count)) output[i] = 0.;
}
void launch_float64_attention_cuda(Float64Attention op, uint32_t* output, const uint32_t* a,
        const uint32_t* b, const uint32_t* c, const uint32_t* d, const KernelParams& p) {
    if (!p.output_count()) return;
    double* destination = reinterpret_cast<double*>(output);
    const double* query = reinterpret_cast<const double*>(a), *key = reinterpret_cast<const double*>(b);
    const cudaStream_t stream = current_cuda_stream();
    if (op == Float64Attention::Softmax) {
        const int axis = p.attribute(0), rank = p.input_rank(KernelInput::A);
        const int length = rank ? p.input_shape(KernelInput::A, axis) : 1, inner = rank ? p.input_stride(KernelInput::A, axis) : 1;
        const int rows = p.output_count() / length;
        float64_attention_softmax_kernel<<<(unsigned(rows) + 127u) / 128u, 128, 0, stream>>>(destination, query, key, rows, length, inner, p);
        return;
    }
    const Float64AttentionShape s(p);
    if (!s.keys) {
        float64_attention_zero_kernel<<<(unsigned(p.output_count()) + 127u) / 128u, 128, 0, stream>>>(destination, p.output_count());
        return;
    }
    const double* values = reinterpret_cast<const double*>(c), *mask = reinterpret_cast<const double*>(d);
    const bool by_head = p.attribute(4) != 0;
    const int tile = by_head ? s.queries : std::min(s.queries, p.attribute(1));
    const int heads = by_head ? 1 : s.batch * s.heads;
    BufferPtr storage = backend_workspace(Backend::CUDA, checked_memory_mul(size_t(heads) * tile * s.keys, sizeof(double)));
    double* scores = static_cast<double*>(storage->raw_pointer());
    LocalFunction execute{[&](int first_head, int start, int rows) {
        const int score_count = heads * rows * s.keys, row_count = heads * rows, value_count = heads * rows * s.values;
        float64_attention_scores_kernel<<<(unsigned(score_count) + 127u) / 128u, 128, 0, stream>>>(scores, query, key, score_count, first_head, start, rows, s);
        float64_attention_probabilities_kernel<<<(unsigned(row_count) + 127u) / 128u, 128, 0, stream>>>(scores, mask, row_count, first_head, start, rows, s, p);
        float64_attention_values_kernel<<<(unsigned(value_count) + 127u) / 128u, 128, 0, stream>>>(destination, scores, values, value_count, first_head, start, rows, s);
    }};
    if (by_head) for (int head = 0; head < s.batch * s.heads; ++head) execute(head, 0, s.queries);
    else for (int start = 0; start < s.queries;) {
        const int rows = std::min(tile, s.queries - start);
        execute(0, start, rows); start += rows;
    }
}
} // namespace elman
