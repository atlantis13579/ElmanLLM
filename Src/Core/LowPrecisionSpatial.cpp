#include "Operators/low_precision_spatial.h"
#include <stdexcept>

namespace elman {
void validate_low_precision_spatial(LowPrecisionSpatial op, const BufferPtr& output,
                              const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool convolution = op <= LowPrecisionSpatial::WeightGradient;
    const bool forward = op == LowPrecisionSpatial::Convolution;
    const bool pool_gradient = op == LowPrecisionSpatial::MaximumPoolGradient || op == LowPrecisionSpatial::AveragePoolGradient;
    const bool bias = forward && inputs.size() > 2 && inputs[2];
    const int arity = convolution ? (bias ? 3 : 2) : (pool_gradient ? 2 : 1);
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid low precision spatial operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected Low precision spatial operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected Low precision spatial operand descriptor");
    }
    require(is_low_precision_float(output->dtype), "Low precision spatial operation requires a Low precision output");
    require(p.output_dtype() == int(output->dtype), "Spatial output descriptor dtype differs");
    auto shape = [&](int rank, auto dimension, auto stride, size_t expected) {
        require(rank >= 0 && rank <= 8, "Invalid low precision spatial rank");
        std::vector<int> result(rank);
        size_t forward_count = 1, reverse_count = 1;
        for (int d = 0; d < rank; ++d) {
            const int length = dimension(d), reverse = dimension(rank - d - 1);
            require(length >= 0 && reverse >= 0 &&
                    (length == 0 || forward_count <= size_t(INT32_MAX) / size_t(length)) &&
                    (reverse == 0 || reverse_count <= size_t(INT32_MAX) / size_t(reverse)), "Low precision spatial dimensions overflow");
            require(stride(rank - d - 1, int(reverse_count)), "Invalid low precision spatial contiguous stride");
            forward_count *= size_t(length); reverse_count *= size_t(reverse); result[d] = length;
        }
        require(forward_count == expected, "Low precision spatial shape differs from storage");
        return result;
    };
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == output->dtype, "Low precision spatial operands must be Low precision");
        const auto input = KernelInput(i);
        require(p.input_dtype(input) == int(inputs[i]->dtype), "Spatial input descriptor dtype differs");
        shapes.push_back(shape(p.input_rank(input), [&](int d) { return p.input_shape(input, d); },
            [&](int d, int value) { return p.input_stride(input, d) == value; }, inputs[i]->count));
        require(shapes.back().size() == size_t(i == 2 ? 1 : 4), "Invalid low precision spatial input shape");
    }
    const auto target = shape(p.output_rank(), [&](int d) { return p.output_shape(d); }, [](int, int) { return true; }, output->count);
    require(target.size() == 4 && p.output_count() == int(output->count), "Invalid low precision spatial output shape/count");
    auto extent = [&](int length, int kernel, int stride, int padding, int dilation) {
        require(kernel > 0 && stride > 0 && padding >= 0 && dilation > 0, "Invalid low precision spatial attributes");
        const int64_t available = int64_t(length) + 2 * int64_t(padding), effective = int64_t(kernel - 1) * dilation + 1;
        require(available >= effective, "Low precision spatial kernel exceeds padded input");
        const int64_t result = (available - effective) / stride + 1;
        require(result <= INT32_MAX, "Low precision spatial output extent overflows");
        return int(result);
    };
    if (convolution) {
        const auto& x = op == LowPrecisionSpatial::InputGradient ? target : shapes[0];
        const auto& w = op == LowPrecisionSpatial::WeightGradient ? target : shapes[1];
        const auto& y = forward ? target : op == LowPrecisionSpatial::InputGradient ? shapes[0] : shapes[1];
        const int groups = p.attribute(6);
        require(groups > 0 && w[0] > 0 && w[0] % groups == 0 && int64_t(w[1]) * groups == x[1], "Invalid low precision convolution channel groups");
        require(x[0] == 0 || x[1] == 0 || (x[2] > 0 && x[3] > 0), "Low precision convolution requires nonempty spatial dimensions");
        require(y[0] == x[0] && y[1] == (x[1] == 0 ? 0 : w[0]), "Low precision convolution batch/output channels differ");
        require(y[2] == extent(x[2], w[2], p.attribute(0), p.attribute(2), p.attribute(4)) &&
                y[3] == extent(x[3], w[3], p.attribute(1), p.attribute(3), p.attribute(5)), "Low precision convolution output dimensions differ");
        if (bias) require(shapes[2][0] == w[0], "Low precision convolution bias shape differs");
    } else {
        const auto& x = shapes[0];
        const auto& y = pool_gradient ? shapes[1] : target;
        require(!pool_gradient || target == x, "Low precision pooling gradient shape differs");
        require(x[1] > 0 && x[2] > 0 && x[3] > 0 && y[0] == x[0] && y[1] == x[1], "Invalid low precision pooling batch/channel dimensions");
        require(p.attribute(4) <= p.attribute(0) / 2 && p.attribute(5) <= p.attribute(1) / 2 &&
                (p.attribute(6) == 0 || p.attribute(6) == 1), "Invalid low precision pooling padding/count flag");
        require(y[2] == extent(x[2], p.attribute(0), p.attribute(2), p.attribute(4), 1) &&
                y[3] == extent(x[3], p.attribute(1), p.attribute(3), p.attribute(5), 1), "Low precision pooling output dimensions differ");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        require(a <= b ? b - a >= output->bytes : a - b >= inputs[i]->bytes, "Low precision spatial output overlaps an input");
    }
}
} // namespace elman
