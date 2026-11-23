#include "Operators/float64_spatial.h"
#include "Core/LocalFunction.h"
#include <optional>
#include <stdexcept>

namespace elman {
void validate_float64_spatial(Float64Spatial op, const BufferPtr& output,
                              const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool convolution = op <= Float64Spatial::WeightGradient;
    const bool forward = op == Float64Spatial::Convolution;
    const bool pool_gradient = op == Float64Spatial::MaximumPoolGradient || op == Float64Spatial::AveragePoolGradient;
    const bool bias = forward && inputs.size() > 2 && inputs[2];
    const int arity = convolution ? (bias ? 3 : 2) : (pool_gradient ? 2 : 1);
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid Double spatial operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected Double spatial operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected Double spatial operand descriptor");
    }
    require(output->dtype == DType::Float64, "Double spatial operation requires a Double output");
    LocalFunction shape{[&](std::optional<KernelInput> input, size_t expected) {
        const int rank = input ? p.input_rank(*input) : p.output_rank();
        require(rank >= 0 && rank <= 8, "Invalid Double spatial rank");
        std::vector<int> result(rank);
        size_t forward_count = 1, reverse_count = 1;
        for (int d = 0; d < rank; ++d) {
            const int reverse_axis = rank - d - 1;
            const int length = input ? p.input_shape(*input, d) : p.output_shape(d);
            const int reverse = input ? p.input_shape(*input, reverse_axis) : p.output_shape(reverse_axis);
            require(length >= 0 && reverse >= 0 &&
                    (length == 0 || forward_count <= size_t(INT32_MAX) / size_t(length)) &&
                    (reverse == 0 || reverse_count <= size_t(INT32_MAX) / size_t(reverse)), "Double spatial dimensions overflow");
            require(!input || p.input_stride(*input, reverse_axis) == int(reverse_count), "Invalid Double spatial contiguous stride");
            forward_count *= size_t(length); reverse_count *= size_t(reverse); result[d] = length;
        }
        require(forward_count == expected, "Double spatial shape differs from storage");
        return result;
    }};
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == DType::Float64, "Double spatial operands must be Double");
        const KernelInput input = KernelInput(i);
        shapes.push_back(shape(input, inputs[i]->count));
        require(shapes.back().size() == size_t(i == 2 ? 1 : 4), "Invalid Double spatial input shape");
    }
    const std::vector<int> target = shape(std::nullopt, output->count);
    require(target.size() == 4 && p.output_count() == int(output->count), "Invalid Double spatial output shape/count");
    LocalFunction extent{[&](int length, int kernel, int stride, int padding, int dilation) {
        require(kernel > 0 && stride > 0 && padding >= 0 && dilation > 0, "Invalid Double spatial attributes");
        const int64_t available = int64_t(length) + 2 * int64_t(padding), effective = int64_t(kernel - 1) * dilation + 1;
        require(available >= effective, "Double spatial kernel exceeds padded input");
        const int64_t result = (available - effective) / stride + 1;
        require(result <= INT32_MAX, "Double spatial output extent overflows");
        return int(result);
    }};
    if (convolution) {
        const std::vector<int>& x = op == Float64Spatial::InputGradient ? target : shapes[0];
        const std::vector<int>& w = op == Float64Spatial::WeightGradient ? target : shapes[1];
        const std::vector<int>& y = forward ? target : op == Float64Spatial::InputGradient ? shapes[0] : shapes[1];
        const int groups = p.attribute(6);
        require(groups > 0 && w[0] > 0 && w[0] % groups == 0 && int64_t(w[1]) * groups == x[1], "Invalid Double convolution channel groups");
        require(x[0] == 0 || x[1] == 0 || (x[2] > 0 && x[3] > 0), "Double convolution requires nonempty spatial dimensions");
        require(y[0] == x[0] && y[1] == (x[1] == 0 ? 0 : w[0]), "Double convolution batch/output channels differ");
        require(y[2] == extent(x[2], w[2], p.attribute(0), p.attribute(2), p.attribute(4)) &&
                y[3] == extent(x[3], w[3], p.attribute(1), p.attribute(3), p.attribute(5)), "Double convolution output dimensions differ");
        if (bias) require(shapes[2][0] == w[0], "Double convolution bias shape differs");
    } else {
        const std::vector<int>& x = shapes[0];
        const std::vector<int>& y = pool_gradient ? shapes[1] : target;
        require(!pool_gradient || target == x, "Double pooling gradient shape differs");
        require(x[1] > 0 && x[2] > 0 && x[3] > 0 && y[0] == x[0] && y[1] == x[1], "Invalid Double pooling batch/channel dimensions");
        require(p.attribute(4) <= p.attribute(0) / 2 && p.attribute(5) <= p.attribute(1) / 2 &&
                (p.attribute(6) == 0 || p.attribute(6) == 1), "Invalid Double pooling padding/count flag");
        require(y[2] == extent(x[2], p.attribute(0), p.attribute(2), p.attribute(4), 1) &&
                y[3] == extent(x[3], p.attribute(1), p.attribute(3), p.attribute(5), 1), "Double pooling output dimensions differ");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        require(a <= b ? b - a >= output->bytes : a - b >= inputs[i]->bytes, "Double spatial output overlaps an input");
    }
}
} // namespace elman
