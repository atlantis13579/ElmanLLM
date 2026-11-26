#include "Kernals/integer_matrix.h"
#include "Core/LocalFunction.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace elman {
void validate_integer_matrix_coefficient(float value, DType dtype) {
    const bool signed_type = integer_signed(int(dtype));
    const unsigned bits = dtype_info(dtype).bits;
    const double bound = std::ldexp(1.0, int(bits) - int(signed_type));
    // The public scalar ABI is Float32. Narrow integer limits are exact; wider
    // positive limits round to the next power of two and must be exclusive.
    const bool upper = bits >= 32 ? double(value) < bound : double(value) <= bound - 1;
    if (!is_standard_integer(dtype) || dtype == DType::Bool || !std::isfinite(value) ||
        double(value) < (signed_type ? -bound : 0.0) || !upper)
        throw std::invalid_argument("Integer matrix coefficient is not representable in the output dtype");
}

void validate_integer_matrix(IntegerMatrix operation, const BufferPtr& output,
                             const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool linear = operation == IntegerMatrix::Linear || operation == IntegerMatrix::PackedLinear;
    const bool packed = operation == IntegerMatrix::PackedLinear;
    const bool addmm = operation == IntegerMatrix::Addmm;
    require(inputs.size() >= 2 && inputs.size() <= 4 && inputs[0] && inputs[1], "Integer matrix requires two inputs");
    require(inputs.size() < 4 || !inputs[3], "Unexpected integer matrix fourth input");
    const bool bias = inputs.size() > 2 && bool(inputs[2]);
    require(!addmm || bias, "Integer addmm requires its addend");
    require(linear || addmm || !bias, "Unexpected integer matrix addend");
    LocalFunction shape_count{[&](const std::vector<int>& shape) {
        size_t forward = 1, backward = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            const int a = shape[i], b = shape[shape.size() - i - 1];
            require(a >= 0 && b >= 0, "Negative integer matrix dimension");
            require((a == 0 || forward <= size_t(INT32_MAX) / size_t(a)) &&
                    (b == 0 || backward <= size_t(INT32_MAX) / size_t(b)), "Integer matrix dimensions exceed Int32 indexing");
            forward *= size_t(a); backward *= size_t(b);
        }
        return forward;
    }};
    std::vector<std::vector<int>> shape(3);
    for (int i = 0; i < 3; ++i) {
        const KernelInput input = KernelInput(i);
        const int rank = p.input_rank(input);
        const bool present = size_t(i) < inputs.size() && bool(inputs[i]);
        if (!present) { require(i == 2 && rank == -1, "Missing integer matrix descriptor"); continue; }
        require(inputs[i]->dtype == output->dtype && rank >= 0 && rank <= 8, "Invalid integer matrix input rank or dtype");
        shape[i].resize(rank);
        for (int d = 0; d < rank; ++d) shape[i][d] = p.input_shape(input, d);
        require(shape_count(shape[i]) == inputs[i]->count, "Integer matrix input shape differs from storage");
        int stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            require(p.input_stride(input, d) == stride, "Invalid integer matrix contiguous stride");
            stride *= shape[i][d];
        }
    }
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8, "Invalid integer matrix output rank");
    std::vector<int> target(rank), expected;
    for (int d = 0; d < rank; ++d) target[d] = p.output_shape(d);
    require(shape_count(target) == output->count && p.output_count() == int(output->count), "Integer matrix output shape differs from storage");
    const std::vector<int>& a = shape[0]; const std::vector<int>& b = shape[1]; const std::vector<int>& c = shape[2];
    if (linear) {
        require(!a.empty() && b.size() == 2 && a.back() == b[1], "Invalid integer linear feature dimensions");
        const int columns = b[0];
        require(!bias || ((!packed && c.empty()) || c == std::vector<int>{columns}), "Invalid integer linear bias shape");
        expected = a;
        expected.back() = columns;
        if (packed) {
            const int sections = p.attribute(0);
            require(sections > 0 && sections < KernelParams::kAttributeCount, "Invalid integer packed projection count");
            int total = 0;
            for (int group = 0; group < sections; ++group) {
                const int width = p.attribute(group + 1);
                require(width >= 0 && width <= columns - total, "Invalid integer packed projection width");
                total += width;
            }
            require(total == columns, "Integer packed projection widths differ");
            expected = {int(shape_count(expected))};
        }
    } else {
        require(a.size() >= 2 && b.size() >= 2, "Integer matrix requires rank at least two");
        if (operation == IntegerMatrix::Bmm)
            require(a.size() == 3 && b.size() == 3 && a[0] == b[0], "Integer bmm requires matching batches");
        if (addmm) require(a.size() == 2 && b.size() == 2, "Integer addmm requires matrices");
        if (operation == IntegerMatrix::Matmul)
            require(p.attribute(0) == 0 || p.attribute(0) == 1, "Invalid integer matrix transpose flag");
        const bool transposed = operation == IntegerMatrix::Matmul && p.attribute(0) != 0;
        require(a.back() == b[b.size() - (transposed ? 1 : 2)], "Integer matrix inner dimensions differ");
        const size_t dimensions = std::max(a.size(), b.size());
        expected.resize(dimensions);
        for (size_t d = 0; d + 2 < dimensions; ++d) {
            const int left = d + a.size() >= dimensions ? a[d + a.size() - dimensions] : 1;
            const int right = d + b.size() >= dimensions ? b[d + b.size() - dimensions] : 1;
            require(left == right || left == 1 || right == 1, "Integer matrix batch dimensions differ");
            expected[d] = left == 1 ? right : left;
        }
        expected[dimensions - 2] = a[a.size() - 2];
        expected.back() = b[b.size() - (transposed ? 2 : 1)];
        if (addmm) {
            require(c.size() <= 2, "Invalid integer addmm addend rank");
            for (size_t d = 0; d < c.size(); ++d)
                require(c[d] == 1 || c[d] == expected[d + 2 - c.size()], "Integer addmm addend does not broadcast");
            for (int slot : {0, 1}) {
                const int bits = p.attribute(slot);
                float value;
                std::memcpy(&value, &bits, sizeof(value));
                validate_integer_matrix_coefficient(value, output->dtype);
            }
        }
    }
    require(target == expected, "Integer matrix output dimensions differ");
    for (const BufferPtr& input : inputs) if (input && output->bytes && input->bytes) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(input->raw_pointer());
        require(a <= b ? b - a >= output->bytes : a - b >= input->bytes, "Integer matrix output overlaps an input");
    }
}
} // namespace elman
