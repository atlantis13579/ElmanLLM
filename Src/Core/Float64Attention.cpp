#include "Operators/float64_attention.h"
#include "Core/Memory.h"
#include <algorithm>
#include <stdexcept>

namespace elman {
void validate_float64_attention(Float64Attention op, const BufferPtr& output,
        const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool tiled = op == Float64Attention::Tiled;
    const int arity = tiled ? 4 : 2;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid Double attention operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected Double attention operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected Double attention operand descriptor");
    }
    require(output->dtype == DType::Float64, "Double attention requires a Double output");
    require(p.attribute(7) == 0 || p.attribute(7) == 1, "Invalid Double attention divisor encoding");
    auto shape = [&](int rank, auto dimension, auto stride, size_t expected) {
        require(rank >= 0 && rank <= 8, "Invalid Double attention rank");
        std::vector<int> result(rank);
        size_t forward = 1, reverse = 1;
        for (int d = 0; d < rank; ++d) {
            const int length = dimension(d), backward = dimension(rank - d - 1);
            require(length >= 0 && backward >= 0 &&
                    (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
                    (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "Double attention dimensions overflow");
            require(stride(rank - d - 1, int(reverse)), "Invalid Double attention contiguous stride");
            forward *= size_t(length); reverse *= size_t(backward); result[d] = length;
        }
        require(forward == expected, "Double attention shape differs from storage");
        return result;
    };
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == DType::Float64, "Double attention requires matching Double operands");
        const auto input = KernelInput(i);
        shapes.push_back(shape(p.input_rank(input), [&](int d) { return p.input_shape(input, d); },
            [&](int d, int value) { return p.input_stride(input, d) == value; }, inputs[i]->count));
    }
    const auto target = shape(p.output_rank(), [&](int d) { return p.output_shape(d); }, [](int, int) { return true; }, output->count);
    require(p.output_count() == int(output->count), "Invalid Double attention output count");
    auto broadcasts = [&](const std::vector<int>& source, const std::vector<int>& destination) {
        require(source.size() <= destination.size(), "Invalid Double attention mask rank");
        for (size_t d = 0; d < source.size(); ++d)
            require(source[d] == 1 || source[d] == destination[destination.size() - source.size() + d], "Double attention mask does not broadcast");
    };
    if (!tiled) {
        require(target == shapes[0], "Double attention softmax output shape differs");
        const int axis = p.attribute(0), rank = int(target.size());
        require(rank ? axis >= 0 && axis < rank : axis == 0, "Invalid Double attention softmax axis");
        require(p.attribute(2) == 0 && p.attribute(3) == 0 && p.attribute(4) == 0, "Causal attributes require tiled attention");
        broadcasts(shapes[1], target);
    } else {
        const auto& q = shapes[0]; const auto& k = shapes[1]; const auto& v = shapes[2];
        require(q.size() == 4 && k.size() == 4 && v.size() == 4 && target.size() == 4, "Double tiled attention requires rank-four tensors");
        require(q[0] == k[0] && q[0] == v[0] && q[1] > 0 && k[1] > 0 && q[1] % k[1] == 0 && k[1] == v[1] &&
                q[3] == k[3] && k[2] == v[2], "Incompatible Double tiled attention dimensions");
        require(target == std::vector<int>({q[0],q[1],q[2],v[3]}), "Double tiled attention output shape differs");
        broadcasts(shapes[3], {q[0],q[1],q[2],k[2]});
        require(p.attribute(1) > 0 && (p.attribute(2) == 0 || p.attribute(2) == 1) &&
                (p.attribute(4) == 0 || p.attribute(4) == 1), "Invalid Double tiled attention attributes");
        size_t elements = p.attribute(4) ? size_t(q[2]) : checked_memory_mul(size_t(q[0]),size_t(q[1]));
        if (!p.attribute(4)) elements = checked_memory_mul(elements,size_t(std::min(q[2],p.attribute(1))));
        elements = checked_memory_mul(elements,size_t(k[2]));
        require(elements <= size_t(INT32_MAX), "Double attention tile exceeds Int32 indexing");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        require(!overlap || (!tiled && i == 0 && a == b), "Unsafe overlapping Double attention operands");
    }
}
} // namespace elman
