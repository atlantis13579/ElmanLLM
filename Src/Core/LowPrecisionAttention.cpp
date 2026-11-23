#include "Operators/low_precision_attention.h"
#include "Core/LocalFunction.h"
#include "Core/Memory.h"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace elman {
void validate_low_precision_attention(LowPrecisionAttention op, const BufferPtr& output,
        const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool tiled = op == LowPrecisionAttention::Tiled;
    const bool matrix = op == LowPrecisionAttention::Matrix;
    const bool reduction = op == LowPrecisionAttention::Sum || op == LowPrecisionAttention::SumTo;
    const bool internal = matrix || reduction || op == LowPrecisionAttention::Gradient || op == LowPrecisionAttention::Probabilities;
    const int arity = tiled ? 4 : (op == LowPrecisionAttention::Probabilities || reduction) ? 1 : 2;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid low precision attention operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected low precision attention operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected low precision attention operand descriptor");
    }
    require(internal ? output->dtype == DType::Float32 : is_low_precision_float(output->dtype), "Invalid attention output dtype");
    require(p.attribute(8) == (op == LowPrecisionAttention::SumTo ? 5 : matrix ? 4 : op == LowPrecisionAttention::Gradient ? 1 : op == LowPrecisionAttention::Probabilities ? 2 : op == LowPrecisionAttention::Sum ? 3 : 0), "Invalid attention intermediate mode");
    require(p.attribute(7) == 0 || p.attribute(7) == 1, "Invalid low precision attention divisor encoding");
    LocalFunction shape{[&](std::optional<KernelInput> input, size_t expected) {
        const int rank = input ? p.input_rank(*input) : p.output_rank();
        require(rank >= 0 && rank <= 8, "Invalid low precision attention rank");
        std::vector<int> result(rank);
        size_t forward = 1, reverse = 1;
        for (int d = 0; d < rank; ++d) {
            const int reverse_axis = rank - d - 1;
            const int length = input ? p.input_shape(*input, d) : p.output_shape(d);
            const int backward = input ? p.input_shape(*input, reverse_axis) : p.output_shape(reverse_axis);
            require(length >= 0 && backward >= 0 &&
                    (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
                    (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "low precision attention dimensions overflow");
            require(!input || p.input_stride(*input, reverse_axis) == int(reverse), "Invalid low precision attention contiguous stride");
            forward *= size_t(length); reverse *= size_t(backward); result[d] = length;
        }
        require(forward == expected, "low precision attention shape differs from storage");
        return result;
    }};
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == output->dtype, "low precision attention requires matching low precision operands");
        const KernelInput input = KernelInput(i);
        shapes.push_back(shape(input, inputs[i]->count));
    }
    const std::vector<int> target = shape(std::nullopt, output->count);
    require(p.output_count() == int(output->count), "Invalid low precision attention output count");
    LocalFunction broadcasts{[&](const std::vector<int>& source, const std::vector<int>& destination) {
        require(source.size() <= destination.size(), "Invalid low precision attention mask rank");
        for (size_t d = 0; d < source.size(); ++d)
            require(source[d] == 1 || source[d] == destination[destination.size() - source.size() + d], "low precision attention mask does not broadcast");
    }};
    if (matrix) {
        require(shapes[0].size() == 4 && shapes[1].size() == 4 && target.size() == 4, "Attention matrix operands require rank four");
        require((p.attribute(0) == 0 || p.attribute(0) == 1) && (p.attribute(1) == 0 || p.attribute(1) == 1), "Invalid attention matrix transpose");
        const std::vector<int>& a = shapes[0]; const std::vector<int>& b = shapes[1];
        require(a[0] == b[0] && a[1] == b[1] && a[p.attribute(0) ? 2 : 3] == b[p.attribute(1) ? 3 : 2], "Incompatible attention matrix operands");
        require(target == std::vector<int>({a[0], a[1], a[p.attribute(0) ? 3 : 2], b[p.attribute(1) ? 2 : 3]}), "Invalid attention matrix output shape");
        for (int i = 2; i < 8; ++i) require(p.attribute(i) == 0, "Unexpected attention matrix attributes");
    } else if (op == LowPrecisionAttention::SumTo) {
        require(target.size() == shapes[0].size() && p.attribute(0) > 0 && p.attribute(0) < (1 << int(target.size())), "Invalid attention reduction axes");
        for (int d = 0; d < int(target.size()); ++d)
            require(target[d] == ((p.attribute(0) & (1 << d)) ? 1 : shapes[0][d]), "Invalid attention reduction output shape");
        for (int i = 1; i < 8; ++i) require(p.attribute(i) == 0, "Unexpected attention reduction attributes");
    } else if (op == LowPrecisionAttention::Sum) {
        require(shapes[0].size() == 2 && target == std::vector<int>({shapes[0][0]}), "Invalid attention reduction shape");
        require(p.attribute(0) == 1, "Invalid attention reduction axis");
        for (int i = 1; i < 8; ++i) require(p.attribute(i) == 0, "Unexpected attention reduction attributes");
    } else if (!tiled) {
        require(target == shapes[0], "low precision attention softmax output shape differs");
        const int axis = p.attribute(0), rank = int(target.size());
        require(rank ? axis >= 0 && axis < rank : axis == 0, "Invalid low precision attention softmax axis");
        require(p.attribute(2) == 0 && p.attribute(3) == 0 && p.attribute(4) == 0, "Causal attributes require tiled attention");
        if (internal) {
            for (int i = 1; i < 8; ++i) require(p.attribute(i) == 0, "Unexpected attention intermediate attributes");
            if (arity == 2) require(shapes[1] == target, "Attention derivative shapes differ");
        }
        else broadcasts(shapes[1], target);
    } else {
        const std::vector<int>& q = shapes[0]; const std::vector<int>& k = shapes[1]; const std::vector<int>& v = shapes[2];
        require(q.size() == 4 && k.size() == 4 && v.size() == 4 && target.size() == 4, "low precision tiled attention requires rank-four tensors");
        require(q[0] == k[0] && q[0] == v[0] && q[1] > 0 && k[1] > 0 && q[1] % k[1] == 0 && k[1] == v[1] &&
                q[3] == k[3] && k[2] == v[2], "Incompatible low precision tiled attention dimensions");
        require(target == std::vector<int>({q[0],q[1],q[2],v[3]}), "low precision tiled attention output shape differs");
        broadcasts(shapes[3], {q[0],q[1],q[2],k[2]});
        require(p.attribute(1) > 0 && (p.attribute(2) == 0 || p.attribute(2) == 1) &&
                (p.attribute(4) == 0 || p.attribute(4) == 1), "Invalid low precision tiled attention attributes");
        size_t elements = p.attribute(4) ? size_t(q[2]) : checked_memory_mul(size_t(q[0]),size_t(q[1]));
        if (!p.attribute(4)) elements = checked_memory_mul(elements,size_t(std::min(q[2],p.attribute(1))));
        elements = checked_memory_mul(elements,size_t(k[2]));
        require(elements <= size_t(INT32_MAX), "low precision attention tile exceeds Int32 indexing");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        require(!overlap || (!internal && !tiled && i == 0 && a == b), "Unsafe overlapping low precision attention operands");
    }
}
} // namespace elman
