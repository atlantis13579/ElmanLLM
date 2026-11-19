#include "Operators/low_precision_loss.h"
#include "Core/Memory.h"
#include <cstring>
#include <stdexcept>

namespace elman {
void validate_low_precision_loss(LowPrecisionLoss op, const BufferPtr& output,
                           const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool mse_gradient = op == LowPrecisionLoss::MeanSquaredErrorGradient;
    const bool mse = op == LowPrecisionLoss::MeanSquaredError || mse_gradient;
    const bool gradient = op == LowPrecisionLoss::CrossEntropyGradient || mse_gradient;
    const int arity = gradient ? 3 : 2;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid low precision loss operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected low precision loss operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected low precision loss operand descriptor");
    }
    require(!inputs.empty() && inputs[0] && is_low_precision_float(inputs[0]->dtype), "Loss requires low precision operands");
    require(output->dtype == inputs[0]->dtype || (mse_gradient && output->dtype == DType::Float32), "Invalid low precision loss output dtype");
    require(p.attribute(0) == int(mse_gradient), "Invalid low precision loss derivative mode");
    auto shape = [&](int rank, auto dimension, auto stride, size_t expected) {
        require(rank >= 0 && rank <= 8, "Invalid low precision loss rank");
        std::vector<int> result(rank);
        size_t forward = 1, reverse = 1;
        for (int d = 0; d < rank; ++d) {
            const int length = dimension(d), backward = dimension(rank - d - 1);
            require(length >= 0 && backward >= 0 &&
                    (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
                    (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "low precision loss dimensions overflow");
            require(stride(rank - d - 1, int(reverse)), "Invalid low precision loss contiguous stride");
            forward *= size_t(length); reverse *= size_t(backward); result[d] = length;
        }
        require(forward == expected, "low precision loss shape differs from storage");
        return result;
    };
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == (!mse && i == 1 ? DType::Int32 : inputs[0]->dtype), "Invalid low precision loss operand dtype");
        const auto input = KernelInput(i);
        shapes.push_back(shape(p.input_rank(input), [&](int d) { return p.input_shape(input, d); },
            [&](int d, int value) { return p.input_stride(input, d) == value; }, inputs[i]->count));
    }
    const auto target = shape(p.output_rank(), [&](int d) { return p.output_shape(d); }, [](int, int) { return true; }, output->count);
    require(p.output_count() == int(output->count), "Invalid low precision loss output count");
    require(gradient ? target == shapes[0] : target.empty(), "Invalid low precision loss output shape");
    if (mse) {
        require(shapes[0] == shapes[1], "Low precision MSE shapes differ");
        require(!mse_gradient || shapes[2].empty(), "MSE requires a scalar upstream gradient");
    }
    else {
        require(shapes[0].size() == 2 && shapes[1].size() == 1 && shapes[0][1] > 0 && shapes[1][0] == shapes[0][0],
                "low precision cross entropy requires rows/classes and one target per row");
        require(!gradient || shapes[2].empty(), "low precision cross entropy requires a scalar upstream gradient");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        require(a <= b ? b - a >= output->bytes : a - b >= inputs[i]->bytes, "low precision loss output overlaps an input");
    }
    if (!mse && inputs[1]->count) {
        MemoryReservation host(Backend::CPU, inputs[1]->bytes);
        const auto raw = backend_download_raw(inputs[1]);
        for (size_t i = 0; i < inputs[1]->count; ++i) {
            int32_t index;
            std::memcpy(&index, raw.data() + i * sizeof(index), sizeof(index));
            require(index >= 0 && index < shapes[0][1], "low precision cross entropy target out of range");
        }
    }
}
} // namespace elman
