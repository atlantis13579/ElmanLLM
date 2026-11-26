#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Kernals/low_precision_loss_math.h"
#include "Kernals/low_precision_softmax.h"
#include "Core/AllocationPool.h"
#include "row_reduction.h"
#include "low_precision_softmax_exp.h"

namespace elman {
void launch_low_precision_loss_cpu(LowPrecisionLoss op, uint32_t* output, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const DType dtype = DType(p.input_dtype(KernelInput::A));
    const int count = low_loss_count(p);
    LocalFunction store{[&](int i, float value) {
        if (p.output_dtype() == int(DType::Float32)) { reinterpret_cast<float*>(output)[i] = value; return; }
        uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
        const uint8_t code = low_precision_encode(value, dtype);
        if (dtype != DType::Float4E2M1FN) bytes[i] = code;
        else bytes[i / 2] = uint8_t((bytes[i / 2] & ~(15 << ((i & 1) * 4))) | (code << ((i & 1) * 4)));
    }};
    if (dtype == DType::Float4E2M1FN && p.output_dtype() != int(DType::Float32)) std::memset(output, 0, (size_t(p.output_count()) + 1) / 2);
    if (op == LowPrecisionLoss::MeanSquaredError) {
        const float total = contiguous_float_sum(count, [&](int i) {
            const float delta = low_precision_value(a, i, dtype) - low_precision_value(b, i, dtype);
            return delta * delta;
        });
        store(0, total / float(count)); return;
    }
    if (op == LowPrecisionLoss::MeanSquaredErrorGradient) {
        for (int i = 0; i < count; ++i) store(i, low_mse_derivative(i, a, b, c, p, count));
        return;
    }
    const int rows = p.input_shape(KernelInput::A, 0), columns = p.input_shape(KernelInput::A, 1);
    if (op == LowPrecisionLoss::CrossEntropy && !rows) { store(0, std::numeric_limits<float>::quiet_NaN()); return; }
    BufferPtr workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(count), sizeof(float)));
    float* values = static_cast<float*>(workspace->raw_pointer());
    launch_low_precision_softmax_cpu(LowPrecisionSoftmax::LogSoftmax, reinterpret_cast<uint32_t*>(values), a, nullptr, low_loss_softmax_params(p));
    const int32_t* ids = reinterpret_cast<const int32_t*>(b);
    if (op == LowPrecisionLoss::CrossEntropy) {
        int logarithm = 0;
        for (unsigned n = rows > 0 ? unsigned(rows - 1) : 0; n; n >>= 1) ++logarithm;
        const int power = std::max(4, logarithm / 8);
        const int64_t mask = (int64_t(1) << power) - 1;
        float partial[8]{};
        for (int row = 0; row < rows; ++row) {
            partial[0] -= values[row * columns + ids[row]];
            for (int level = 0; level < 7; ++level) {
                if (int64_t(row) & (mask << (level * power))) break;
                partial[level + 1] += partial[level]; partial[level] = 0;
            }
        }
        float total = 0;
        for (float value : partial) total += value;
        store(0, total / float(rows)); return;
    }
    if (!count) return;
    const float selected = -low_precision_value(c, 0, dtype) / float(rows);
    const float total = columns == 1 ? selected : selected + 0.f;
    for (int i = 0; i < count; ++i) {
        const float gradient = i % columns == ids[i / columns] ? selected : 0.f;
        store(i, gradient - low_softmax_vector_exp(values[i]) * total);
    }
}
} // namespace elman
