#include "kernel.h"
#include "row_reduction.h"
#include "Operators/low_precision_attention.h"
#include "Operators/low_attention_reduction.h"
namespace elman {
namespace {
template<class Read>
float hierarchy(int count, Read read) {
    int logarithm = 0;
    for (int value = count - 1; value > 0; value >>= 1) ++logarithm;
    const int power = std::max(4, logarithm / 4), block = 1 << power;
    float partial[4]{}; int j = 0;
    while (j + block <= count) {
        for (int k = 0; k < block; ++k) partial[0] += read(j++);
        for (int level = 1; level < 4; ++level) {
            partial[level] += partial[level - 1]; partial[level - 1] = 0;
            if (int64_t(j) & (int64_t(block - 1) << (level * power))) break;
        }
    }
    for (; j < count; ++j) partial[0] += read(j);
    return ((partial[0] + partial[1]) + partial[2]) + partial[3];
}
template<class Read>
float scalar_sum(int count, Read read) {
    float partial[4]{};
    for (int lane = 0; lane < 4; ++lane)
        partial[lane] = hierarchy(count / 4, [&](int j) { return read(j * 4 + lane); });
    for (int j = count / 4 * 4; j < count; ++j) partial[0] += read(j);
    return ((partial[0] + partial[1]) + partial[2]) + partial[3];
}
}
void launch_low_attention_matrix_cpu(uint32_t* output, const uint32_t* a, const uint32_t* b, const KernelParams& p) {
    const int rows = p.output_shape(2), columns = p.output_shape(3), shared = p.input_shape(KernelInput::A, p.attribute(0) ? 2 : 3);
    const bool fused = int64_t(rows) * columns * shared >= 400;
    for (int i = 0; i < p.output_count(); ++i) {
        const int batch = i / (rows * columns), row = i / columns % rows, column = i % columns;
        const auto* left = reinterpret_cast<const float*>(a) + batch * rows * shared;
        const auto* right = reinterpret_cast<const float*>(b) + batch * shared * columns;
        // Use the AVX2 GEMM tile width and paired edge accumulators.
        // The one/two-row edges round each product before the pairwise sum.
        const int tail = rows % 6;
        const bool row_tail = row >= rows / 6 * 6;
        const bool pair = fused && (columns <= 8 || (columns % 16 <= 8 && column >= columns / 16 * 16) || (row_tail && tail <= 3 && rows != 1));
        const bool pair_fma = !(row_tail && tail <= 2 && rows != 1);
        float partial[2]{};
        const int first = pair ? shared / 2 * 2 : shared;
        for (int k = 0; k < first; ++k) {
            const float av = left[p.attribute(0) ? k * rows + row : row * shared + k];
            const float bv = right[p.attribute(1) ? column * shared + k : k * columns + column];
            if (pair) {
                if (pair_fma) partial[k % 2] = std::fma(av, bv, partial[k % 2]);
                else partial[k % 2] += av * bv;
            } else if (fused) partial[0] = std::fma(av, bv, partial[0]);
            else partial[0] += av * bv;
        }
        float value = partial[0] + partial[1];
        for (int k = first; k < shared; ++k) {
            const float av = left[p.attribute(0) ? k * rows + row : row * shared + k], bv = right[p.attribute(1) ? column * shared + k : k * columns + column];
            if (pair_fma) value = std::fma(av, bv, value); else value += av * bv;
        }
        reinterpret_cast<float*>(output)[i] = value;
    }
}
void launch_low_attention_sum_cpu(uint32_t* output, const uint32_t* input, const KernelParams& p) {
    const AttentionReduction r(p);
    for (int row = 0; row < p.output_count(); ++row) {
        float value = 0;
        for (int outer = 0; r.block && outer < r.length / r.block; ++outer) {
            auto read = [&](int j) { return reinterpret_cast<const float*>(input)[r.offset(row, outer * r.block + j)]; };
            float partial;
            if (r.fastest && r.block >= 8) partial = contiguous_float_sum(r.block, read);
            else {
                const int width = r.inner >= 8 ? 32 : 4;
                partial = row % r.inner < r.inner / width * width ? hierarchy(r.block, read) : scalar_sum(r.block, read);
            }
            value += partial;
        }
        reinterpret_cast<float*>(output)[row] = value;
    }
}
} // namespace elman
