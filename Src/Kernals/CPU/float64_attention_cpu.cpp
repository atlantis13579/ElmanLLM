#include "Kernals/float64_attention_math.h"
#include "Core/LocalFunction.h"
#include "Core/AllocationPool.h"
#include <algorithm>

namespace elman {
void launch_float64_attention_cpu(Float64Attention op, uint32_t* output, const uint32_t* a,
        const uint32_t* b, const uint32_t* c, const uint32_t* d, const KernelParams& p) {
    if (!p.output_count()) return;
    double* destination = reinterpret_cast<double*>(output);
    const double* query = reinterpret_cast<const double*>(a), *key = reinterpret_cast<const double*>(b);
    if (op == Float64Attention::Softmax) {
        const int axis = p.attribute(0), rank = p.input_rank(KernelInput::A);
        const int length = rank ? p.input_shape(KernelInput::A, axis) : 1, inner = rank ? p.input_stride(KernelInput::A, axis) : 1;
        for (int row = 0; row < p.output_count() / length; ++row)
            float64_attention_softmax_row(destination, query, key, row, length, inner, p);
        return;
    }
    const Float64AttentionShape s(p);
    if (!s.keys) { std::fill(destination, destination + p.output_count(), 0.); return; }
    const double* values = reinterpret_cast<const double*>(c), *mask = reinterpret_cast<const double*>(d);
    const bool by_head = p.attribute(4) != 0;
    const int tile = by_head ? s.queries : std::min(s.queries, p.attribute(1));
    const int heads = by_head ? 1 : s.batch * s.heads;
    BufferPtr storage = backend_workspace(Backend::CPU, checked_memory_mul(size_t(heads) * tile * s.keys, sizeof(double)));
    double* scores = static_cast<double*>(storage->raw_pointer());
    LocalFunction execute{[&](int first_head, int start, int rows) {
        for (int head = 0; head < heads; ++head) for (int row = 0; row < rows; ++row) {
            double* probabilities = scores + (head * rows + row) * s.keys;
            for (int k = 0; k < s.keys; ++k)
                probabilities[k] = float64_attention_score(query, key, first_head + head, start + row, k, s);
            float64_attention_probabilities(probabilities, mask, first_head + head, start + row, s, p);
            for (int column = 0; column < s.values; ++column)
                destination[((first_head + head) * s.queries + start + row) * s.values + column] =
                    float64_attention_value(probabilities, values, first_head + head, column, s);
        }
    }};
    if (by_head) for (int head = 0; head < s.batch * s.heads; ++head) execute(head, 0, s.queries);
    else for (int start = 0; start < s.queries;) {
        const int rows = std::min(tile, s.queries - start);
        execute(0, start, rows); start += rows;
    }
}
} // namespace elman
