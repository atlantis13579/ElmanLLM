#include "Operators/float64_matrix.h"
#include <algorithm>
#include <stdexcept>

namespace elman {
void validate_float64_matrix(Float64Matrix operation, const BufferPtr& output,
                             const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(output->dtype == DType::Float64, "Float64 matrix requires a Double output");
    const bool linear = operation == Float64Matrix::Linear || operation == Float64Matrix::PackedLinear;
    const bool packed = operation == Float64Matrix::PackedLinear;
    const bool addmm = operation == Float64Matrix::Addmm;
    require(inputs.size() >= 2 && inputs.size() <= 4 && inputs[0] && inputs[1], "Float64 matrix requires two inputs");
    require(inputs.size() < 4 || !inputs[3], "Unexpected Float64 matrix fourth input");
    const bool bias = inputs.size() > 2 && bool(inputs[2]);
    require(!addmm || bias, "Float64 addmm requires its addend");
    require(linear || addmm || !bias, "Unexpected Float64 matrix addend");
    auto shape_count = [&](const std::vector<int>& shape) {
        size_t forward = 1, backward = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            const int a = shape[i], b = shape[shape.size() - i - 1];
            require(a >= 0 && b >= 0, "Negative Float64 matrix dimension");
            require((a == 0 || forward <= size_t(INT32_MAX) / size_t(a)) &&
                    (b == 0 || backward <= size_t(INT32_MAX) / size_t(b)), "Float64 matrix dimensions exceed Int32 indexing");
            forward *= size_t(a); backward *= size_t(b);
        }
        return forward;
    };
    std::vector<std::vector<int>> shape(3);
    for (int i = 0; i < 3; ++i) {
        const auto input = KernelInput(i);
        const int rank = p.input_rank(input);
        const bool present = size_t(i) < inputs.size() && bool(inputs[i]);
        if (!present) { require(i == 2 && rank == -1, "Missing Float64 matrix descriptor"); continue; }
        require(inputs[i]->dtype == output->dtype && rank >= 0 && rank <= 8, "Invalid Float64 matrix input rank or dtype");
        shape[i].resize(rank);
        for (int d = 0; d < rank; ++d) shape[i][d] = p.input_shape(input, d);
        require(shape_count(shape[i]) == inputs[i]->count, "Float64 matrix input shape differs from storage");
        int stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            require(p.input_stride(input, d) == stride, "Invalid Float64 matrix contiguous stride");
            stride *= shape[i][d];
        }
    }
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8, "Invalid Float64 matrix output rank");
    std::vector<int> target(rank), expected;
    for (int d = 0; d < rank; ++d) target[d] = p.output_shape(d);
    require(shape_count(target) == output->count && p.output_count() == int(output->count), "Float64 matrix output shape differs from storage");
    const auto& a = shape[0]; const auto& b = shape[1]; const auto& c = shape[2];
    if (linear) {
        require(!a.empty() && b.size() == 2 && a.back() == b[1], "Invalid Float64 linear feature dimensions");
        const int columns = b[0];
        require(!bias || ((!packed && c.empty()) || c == std::vector<int>{columns}), "Invalid Float64 linear bias shape");
        expected = a;
        expected.back() = columns;
        if (packed) {
            const int sections = p.attribute(0);
            require(sections > 0 && sections < KernelParams::kAttributeCount, "Invalid Float64 packed projection count");
            int total = 0;
            for (int group = 0; group < sections; ++group) {
                const int width = p.attribute(group + 1);
                require(width >= 0 && width <= columns - total, "Invalid Float64 packed projection width");
                total += width;
            }
            require(total == columns, "Float64 packed projection widths differ");
            expected = {int(shape_count(expected))};
        }
    } else {
        require(a.size() >= 2 && b.size() >= 2, "Float64 matrix requires rank at least two");
        if (operation == Float64Matrix::Bmm)
            require(a.size() == 3 && b.size() == 3 && a[0] == b[0], "Float64 bmm requires matching batches");
        if (addmm) require(a.size() == 2 && b.size() == 2, "Float64 addmm requires matrices");
        if (operation == Float64Matrix::Matmul)
            require(p.attribute(0) == 0 || p.attribute(0) == 1, "Invalid Float64 matrix transpose flag");
        const bool transposed = operation == Float64Matrix::Matmul && p.attribute(0) != 0;
        require(a.back() == b[b.size() - (transposed ? 1 : 2)], "Float64 matrix inner dimensions differ");
        const size_t dimensions = std::max(a.size(), b.size());
        expected.resize(dimensions);
        for (size_t d = 0; d + 2 < dimensions; ++d) {
            const int left = d + a.size() >= dimensions ? a[d + a.size() - dimensions] : 1;
            const int right = d + b.size() >= dimensions ? b[d + b.size() - dimensions] : 1;
            require(left == right || left == 1 || right == 1, "Float64 matrix batch dimensions differ");
            expected[d] = left == 1 ? right : left;
        }
        expected[dimensions - 2] = a[a.size() - 2];
        expected.back() = b[b.size() - (transposed ? 2 : 1)];
        if (addmm) {
            require(c.size() <= 2, "Invalid Float64 addmm addend rank");
            for (size_t d = 0; d < c.size(); ++d)
                require(c[d] == 1 || c[d] == expected[d + 2 - c.size()], "Float64 addmm addend does not broadcast");
        }
    }
    require(target == expected, "Float64 matrix output dimensions differ");
    for (const auto& input : inputs) if (input && output->bytes && input->bytes) {
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(input->raw_pointer());
        require(a <= b ? b - a >= output->bytes : a - b >= input->bytes, "Float64 matrix output overlaps an input");
    }
}
} // namespace elman
