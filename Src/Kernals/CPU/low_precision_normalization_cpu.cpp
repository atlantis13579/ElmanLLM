#include "kernel.h"
#include "Backends/CpuMath.h"
#include "row_reduction.h"
#include "Kernals/low_precision_normalization_math.h"
#include <vector>
namespace elman {
void launch_low_precision_normalization_cpu(LowPrecisionNormalization op, uint32_t* output, const uint32_t* a,
                                            const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (!p.output_count()) return;
    LowNormLayout layout(op, p);
    std::vector<float> stats(size_t(layout.rows) * 2), coefficients(size_t(layout.rows) * 2), result(p.output_count());
    if (layout.width && (layout.operation != LowPrecisionNormalization::Batch || layout.training)) for (int row = 0; row < layout.rows; ++row) {
        low_norm_statistics(a, stats.data(), row, layout);
        if (layout.mode == 1) low_norm_coefficients(a, b, c, stats.data(), coefficients.data(), row, layout);
    }
    if (layout.dtype == DType::Float32 && layout.operation == LowPrecisionNormalization::Layer && layout.mode >= 2 && layout.rows) {
        const int partitions = std::min(layout.rows, cpu_math_thread_count());
        const int chunk = layout.rows / partitions + int(layout.rows % partitions != 0);
        for (int column = 0; column < layout.width; ++column) {
            float total = 0;
            for (int partition = 0; partition < partitions; ++partition) {
                float partial = 0;
                const int64_t begin = int64_t(partition) * chunk;
                const int64_t end = std::min(int64_t(layout.rows), begin + chunk);
                for (int64_t row = begin; row < end; ++row) {
                    const int index = int(row) * layout.width + column;
                    const float upstream = low_norm_value(b, index, layout);
                    const float inverse = stats[size_t(row) * 2 + 1];
                    const float normalized = low_norm_value(a, index, layout) * inverse + (-inverse * stats[size_t(row) * 2]);
                    partial += layout.mode == 3 ? upstream : upstream * normalized;
                }
                total += partial;
            }
            result[column] = total;
        }
    } else for (int i = 0; i < p.output_count(); ++i)
        result[i] = low_norm_result(i, a, b, c, stats.data(), coefficients.data(), layout);
    if (layout.dtype == DType::Float32) {
        std::memcpy(output, result.data(), result.size() * sizeof(float));
        return;
    }
    uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
    if (layout.dtype == DType::Float4E2M1FN) {
        for (int i = 0; i < p.output_count(); i += 2) bytes[i / 2] = uint8_t(low_precision_encode(result[i], layout.dtype) |
            (i + 1 < p.output_count() ? low_precision_encode(result[i + 1], layout.dtype) << 4 : 0));
    } else for (int i = 0; i < p.output_count(); ++i) bytes[i] = low_precision_encode(result[i], layout.dtype);
}
} // namespace elman
