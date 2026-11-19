#include "Operators/float64_loss.h"
#include "Core/Memory.h"
#include <cstring>
#include <stdexcept>

namespace elman {
void validate_float64_loss(Float64Loss op, const BufferPtr& output,
                           const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool mse = op == Float64Loss::MeanSquaredError, gradient = op == Float64Loss::CrossEntropyGradient;
    const int arity = gradient ? 3 : 2;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid Double loss operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected Double loss operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected Double loss operand descriptor");
    }
    require(output->dtype == DType::Float64, "Double loss requires a Double output");
    auto shape = [&](int rank, auto dimension, auto stride, size_t expected) {
        require(rank >= 0 && rank <= 8, "Invalid Double loss rank");
        std::vector<int> result(rank);
        size_t forward = 1, reverse = 1;
        for (int d = 0; d < rank; ++d) {
            const int length = dimension(d), backward = dimension(rank - d - 1);
            require(length >= 0 && backward >= 0 &&
                    (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
                    (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "Double loss dimensions overflow");
            require(stride(rank - d - 1, int(reverse)), "Invalid Double loss contiguous stride");
            forward *= size_t(length); reverse *= size_t(backward); result[d] = length;
        }
        require(forward == expected, "Double loss shape differs from storage");
        return result;
    };
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == (!mse && i == 1 ? DType::Int32 : DType::Float64), "Invalid Double loss operand dtype");
        const auto input = KernelInput(i);
        shapes.push_back(shape(p.input_rank(input), [&](int d) { return p.input_shape(input, d); },
            [&](int d, int value) { return p.input_stride(input, d) == value; }, inputs[i]->count));
    }
    const auto target = shape(p.output_rank(), [&](int d) { return p.output_shape(d); }, [](int, int) { return true; }, output->count);
    require(p.output_count() == int(output->count), "Invalid Double loss output count");
    require(gradient ? target == shapes[0] : target.empty(), "Invalid Double loss output shape");
    if (mse) require(shapes[0] == shapes[1], "Double MSE shapes differ");
    else {
        require(shapes[0].size() == 2 && shapes[1].size() == 1 && shapes[0][1] > 0 && shapes[1][0] == shapes[0][0],
                "Double cross entropy requires rows/classes and one target per row");
        require(!gradient || shapes[2].empty(), "Double cross entropy requires a scalar upstream gradient");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        require(a <= b ? b - a >= output->bytes : a - b >= inputs[i]->bytes, "Double loss output overlaps an input");
    }
    if (!mse && inputs[1]->count) {
        MemoryReservation host(Backend::CPU, inputs[1]->bytes);
        const auto raw = backend_download_raw(inputs[1]);
        for (size_t i = 0; i < inputs[1]->count; ++i) {
            int32_t index;
            std::memcpy(&index, raw.data() + i * sizeof(index), sizeof(index));
            require(index >= 0 && index < shapes[0][1], "Double cross entropy target out of range");
        }
    }
}
} // namespace elman
