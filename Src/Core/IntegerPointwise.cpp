#include "Operators/integer_pointwise.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace elman {
void validate_integer_pointwise(IntegerPointwise op, const BufferPtr& output,
                                const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    auto numeric = [](DType type) { return is_standard_integer(type) || is_compute_float(type) || type == DType::Float64; };
    const int arity = integer_pointwise_arity(op);
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid integer pointwise operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected integer pointwise operand");
    std::vector<int> shape;
    for (int i = 0; i < arity; ++i) {
        require(bool(inputs[i]), "Missing integer pointwise operand");
        const auto input = KernelInput(i);
        const int rank = p.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid integer pointwise input rank");
        size_t count = 1, prefix = 1;
        std::vector<int> source(rank);
        for (int d = rank - 1; d >= 0; --d) {
            source[d] = p.input_shape(input, d);
            require(source[d] >= 0 && count <= size_t(INT32_MAX) && p.input_stride(input, d) == int(count),
                    "Invalid integer pointwise contiguous descriptor");
            require(source[d] == 0 || count <= size_t(INT32_MAX) / size_t(source[d]),
                    "Integer pointwise dimensions exceed Int32 indexing");
            count *= size_t(source[d]);
        }
        for (int length : source) {
            require(length == 0 || prefix <= size_t(INT32_MAX) / size_t(length),
                    "Integer pointwise prefix dimensions exceed Int32 indexing");
            prefix *= size_t(length);
        }
        require(count == inputs[i]->count, "Integer pointwise input shape differs from storage");
        std::vector<int> combined(std::max(shape.size(), source.size()), 1);
        for (size_t j = 0; j < combined.size(); ++j) {
            const int x = j < shape.size() ? shape[shape.size() - 1 - j] : 1;
            const int y = j < source.size() ? source[source.size() - 1 - j] : 1;
            require(x == y || x == 1 || y == 1, "Integer pointwise inputs cannot broadcast");
            combined[combined.size() - 1 - j] = x == 1 ? y : x;
        }
        shape = std::move(combined);
    }
    require(p.output_rank() == int(shape.size()), "Invalid integer pointwise output rank");
    size_t count = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        require(p.output_shape(int(d)) == shape[d], "Invalid integer pointwise output shape");
        require(shape[d] == 0 || count <= size_t(INT32_MAX) / size_t(shape[d]), "Integer pointwise output exceeds Int32 indexing");
        count *= size_t(shape[d]);
    }
    require(count == output->count && p.output_count() == int(count), "Integer pointwise output shape differs from storage");
    const DType a = inputs[0]->dtype, out = output->dtype;
    const bool logical = op == IntegerPointwise::LogicalAnd || op == IntegerPointwise::LogicalOr || op == IntegerPointwise::LogicalNot;
    const bool compare = op >= IntegerPointwise::Eq && op <= IntegerPointwise::Le;
    if (op == IntegerPointwise::Cast) {
        require(numeric(a) && numeric(out) && (is_standard_integer(a) || is_standard_integer(out)), "Unsupported integer cast dtype");
    } else if (logical) {
        require(out == DType::Bool, "Logical output must be Bool");
        for (int i = 0; i < arity; ++i) require(numeric(inputs[i]->dtype), "Unsupported logical operand dtype");
    } else if (op == IntegerPointwise::Where) {
        require(a == DType::Bool && is_standard_integer(out) && inputs[1]->dtype == out && inputs[2]->dtype == out,
                "Invalid integer where dtypes");
    } else {
        require(is_standard_integer(a) && out == (compare ? DType::Bool : a), "Invalid integer pointwise output dtype");
        if (arity == 2) require(inputs[1]->dtype == a, "Integer pointwise operands must have the same promoted dtype");
        require(a != DType::Bool || (op != IntegerPointwise::Sub && op != IntegerPointwise::Pow &&
                (op != IntegerPointwise::Abs || output->backend == Backend::CUDA) && op != IntegerPointwise::Relu), "Unsupported Bool arithmetic");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto first = reinterpret_cast<uintptr_t>(output->raw_pointer());
        const auto second = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = first <= second ? second - first < output->bytes : first - second < inputs[i]->bytes;
        if (!overlap) continue;
        require(first == second && out == inputs[i]->dtype && output->count == inputs[i]->count &&
                p.output_rank() == p.input_rank(KernelInput(i)), "Unsafe overlapping integer pointwise buffers");
        for (size_t d = 0; d < shape.size(); ++d)
            require(shape[d] == p.input_shape(KernelInput(i), int(d)), "Overlapping integer broadcast operand");
    }
}
} // namespace elman
