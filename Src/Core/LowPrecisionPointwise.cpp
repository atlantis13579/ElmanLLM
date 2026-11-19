#include "Operators/low_precision_pointwise.h"
#include <algorithm>
#include <stdexcept>

namespace elman {
void validate_low_precision_pointwise(LowPrecisionPointwise op, const BufferPtr& output,
                                      const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const int arity = low_precision_pointwise_arity(op);
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid low precision pointwise operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected low precision pointwise operand");
    for (int i = arity; i < 4; ++i) require(p.input_rank(KernelInput(i)) == -1, "Unexpected low precision pointwise descriptor");
    DType dtype = output->dtype;
    if (low_precision_pointwise_boolean(op)) {
        require(output->dtype == DType::Bool && inputs[0], "Low precision comparison requires Bool output");
        dtype = inputs[0]->dtype;
    }
    require(is_low_precision_float(dtype), "Low precision pointwise output dtype is invalid");
    std::vector<int> shape;
    for (int i = 0; i < arity; ++i) {
        require(bool(inputs[i]), "Missing low precision pointwise operand");
        const auto input = KernelInput(i);
        const int rank = p.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid low precision pointwise input rank");
        const bool scalar = arity == 2 && !low_precision_pointwise_boolean(op) &&
            inputs[i]->dtype == DType::Float32 && rank == 0;
        require(op == LowPrecisionPointwise::Where && i == 0 ? inputs[i]->dtype == DType::Bool :
                inputs[i]->dtype == dtype || scalar, "Low precision pointwise input dtype is invalid");
        size_t count = 1, prefix = 1;
        std::vector<int> source(rank);
        for (int d = rank - 1; d >= 0; --d) {
            source[d] = p.input_shape(input, d);
            require(source[d] >= 0 && count <= size_t(INT32_MAX) && p.input_stride(input, d) == int(count),
                    "Invalid low precision pointwise contiguous descriptor");
            require(!source[d] || count <= size_t(INT32_MAX) / size_t(source[d]), "Low precision pointwise dimensions exceed Int32 indexing");
            count *= size_t(source[d]);
        }
        for (int length : source) {
            require(!length || prefix <= size_t(INT32_MAX) / size_t(length), "Low precision pointwise prefix exceeds Int32 indexing");
            prefix *= size_t(length);
        }
        require(count == inputs[i]->count, "Low precision pointwise input shape differs from storage");
        std::vector<int> combined(std::max(shape.size(), source.size()), 1);
        for (size_t j = 0; j < combined.size(); ++j) {
            const int a = j < shape.size() ? shape[shape.size() - 1 - j] : 1;
            const int b = j < source.size() ? source[source.size() - 1 - j] : 1;
            require(a == b || a == 1 || b == 1, "Low precision operands cannot broadcast");
            combined[combined.size() - 1 - j] = a == 1 ? b : a;
        }
        shape = std::move(combined);
    }
    require(p.output_rank() == int(shape.size()), "Low precision pointwise output rank differs");
    size_t count = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        require(p.output_shape(int(d)) == shape[d], "Low precision pointwise output shape differs");
        require(!shape[d] || count <= size_t(INT32_MAX) / size_t(shape[d]), "Low precision output exceeds Int32 indexing");
        count *= size_t(shape[d]);
    }
    require(count == output->count && p.output_count() == int(count), "Low precision pointwise output differs from storage");
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer());
        const auto b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        if (!overlap) continue;
        require(a == b && output->dtype == inputs[i]->dtype && output->count == inputs[i]->count &&
                p.output_rank() == p.input_rank(KernelInput(i)), "Unsafe overlapping low precision pointwise buffers");
        for (size_t d = 0; d < shape.size(); ++d)
            require(shape[d] == p.input_shape(KernelInput(i), int(d)), "Overlapping low precision broadcast operand");
    }
}
} // namespace elman
