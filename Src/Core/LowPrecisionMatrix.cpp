#include "Operators/low_precision_matrix.h"
#include "Core/LocalFunction.h"
#include <algorithm>
#include <stdexcept>

namespace elman {
void validate_low_precision_matrix(LowPrecisionMatrix operation, const BufferPtr& output,
                             const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(is_low_precision_float(output->dtype), "Low precision matrix requires a low precision output");
    const bool linear = operation == LowPrecisionMatrix::Linear || operation == LowPrecisionMatrix::PackedLinear;
    const bool packed = operation == LowPrecisionMatrix::PackedLinear;
    const bool addmm = operation == LowPrecisionMatrix::Addmm;
    require(inputs.size() >= 2 && inputs.size() <= 4 && inputs[0] && inputs[1], "Low precision matrix requires two inputs");
    require(p.input_rank(KernelInput::D) == -1 && (inputs.size() < 4 || !inputs[3]), "Unexpected Low precision matrix fourth input");
    const bool bias = inputs.size() > 2 && bool(inputs[2]);
    require(!addmm || bias, "Low precision addmm requires its addend");
    require(linear || addmm || !bias, "Unexpected Low precision matrix addend");
    LocalFunction shape_count{[&](const std::vector<int>& shape) {
        size_t forward = 1, backward = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            const int a = shape[i], b = shape[shape.size() - i - 1];
            require(a >= 0 && b >= 0, "Negative Low precision matrix dimension");
            require((a == 0 || forward <= size_t(INT32_MAX) / size_t(a)) &&
                    (b == 0 || backward <= size_t(INT32_MAX) / size_t(b)), "Low precision matrix dimensions exceed Int32 indexing");
            forward *= size_t(a); backward *= size_t(b);
        }
        return forward;
    }};
    std::vector<std::vector<int>> shape(3);
    for (int i = 0; i < 3; ++i) {
        const KernelInput input = KernelInput(i);
        const int rank = p.input_rank(input);
        const bool present = size_t(i) < inputs.size() && bool(inputs[i]);
        if (!present) { require(i == 2 && rank == -1, "Missing Low precision matrix descriptor"); continue; }
        require(inputs[i]->dtype == output->dtype && rank >= 0 && rank <= 8, "Invalid Low precision matrix input rank or dtype");
        shape[i].resize(rank);
        for (int d = 0; d < rank; ++d) shape[i][d] = p.input_shape(input, d);
        require(shape_count(shape[i]) == inputs[i]->count, "Low precision matrix input shape differs from storage");
        int stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            require(p.input_stride(input, d) == stride, "Invalid Low precision matrix contiguous stride");
            stride *= shape[i][d];
        }
    }
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8, "Invalid Low precision matrix output rank");
    std::vector<int> target(rank), expected;
    for (int d = 0; d < rank; ++d) target[d] = p.output_shape(d);
    require(shape_count(target) == output->count && p.output_count() == int(output->count), "Low precision matrix output shape differs from storage");
    const std::vector<int>& a = shape[0]; const std::vector<int>& b = shape[1]; const std::vector<int>& c = shape[2];
    if (linear) {
        require(!a.empty() && b.size() == 2 && a.back() == b[1], "Invalid Low precision linear feature dimensions");
        const int columns = b[0];
        require(!bias || ((!packed && c.empty()) || c == std::vector<int>{columns}), "Invalid Low precision linear bias shape");
        expected = a;
        expected.back() = columns;
        if (packed) {
            const int sections = p.attribute(0);
            require(sections > 0 && sections < KernelParams::kAttributeCount, "Invalid Low precision packed projection count");
            int total = 0;
            for (int group = 0; group < sections; ++group) {
                const int width = p.attribute(group + 1);
                require(width >= 0 && width <= columns - total, "Invalid Low precision packed projection width");
                total += width;
            }
            require(total == columns, "Low precision packed projection widths differ");
            expected = {int(shape_count(expected))};
        }
    } else {
        require(a.size() >= 2 && b.size() >= 2, "Low precision matrix requires rank at least two");
        if (operation == LowPrecisionMatrix::Bmm)
            require(a.size() == 3 && b.size() == 3 && a[0] == b[0], "Low precision bmm requires matching batches");
        if (addmm) require(a.size() == 2 && b.size() == 2, "Low precision addmm requires matrices");
        if (operation == LowPrecisionMatrix::Matmul)
            require((p.attribute(0) == 0 || p.attribute(0) == 1) && (p.attribute(1) == 0 || p.attribute(1) == 1),
                    "Invalid low precision matrix transpose or folding flag");
        const bool transposed = operation == LowPrecisionMatrix::Matmul && p.attribute(0) != 0;
        require(a.back() == b[b.size() - (transposed ? 1 : 2)], "Low precision matrix inner dimensions differ");
        const size_t dimensions = std::max(a.size(), b.size());
        expected.resize(dimensions);
        for (size_t d = 0; d + 2 < dimensions; ++d) {
            const int left = d + a.size() >= dimensions ? a[d + a.size() - dimensions] : 1;
            const int right = d + b.size() >= dimensions ? b[d + b.size() - dimensions] : 1;
            require(left == right || left == 1 || right == 1, "Low precision matrix batch dimensions differ");
            expected[d] = left == 1 ? right : left;
        }
        expected[dimensions - 2] = a[a.size() - 2];
        expected.back() = b[b.size() - (transposed ? 2 : 1)];
        if (addmm) {
            require(c.size() <= 2, "Invalid Low precision addmm addend rank");
            for (size_t d = 0; d < c.size(); ++d)
                require(c[d] == 1 || c[d] == expected[d + 2 - c.size()], "Low precision addmm addend does not broadcast");
        }
    }
    require(target == expected, "Low precision matrix output dimensions differ");
    for (const BufferPtr& input : inputs) if (input && output->bytes && input->bytes) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(input->raw_pointer());
        require(a <= b ? b - a >= output->bytes : a - b >= input->bytes, "Low precision matrix output overlaps an input");
    }
}
} // namespace elman
