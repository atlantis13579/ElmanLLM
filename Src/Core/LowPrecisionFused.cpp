#include "Operators/low_precision_fused.h"
#include <algorithm>
#include <stdexcept>

namespace elman {
void validate_low_precision_fused(LowPrecisionFused op, const BufferPtr& output,
                                      const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const int arity = op == LowPrecisionFused::Rotary ? 3 : 2;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid low precision fused operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected low precision fused operand");
    for (int i = arity; i < 4; ++i) require(p.input_rank(KernelInput(i)) == -1, "Unexpected low precision fused descriptor");
    const DType dtype = output->dtype;
    require(is_low_precision_float(dtype), "Low precision fused output dtype is invalid");
    std::vector<int> shape;
    for (int i = 0; i < arity; ++i) {
        require(bool(inputs[i]), "Missing low precision fused operand");
        const auto input = KernelInput(i);
        const int rank = p.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid low precision fused input rank");
        require(inputs[i]->dtype == dtype, "Low precision fused operands must match");
        size_t count = 1, prefix = 1;
        std::vector<int> source(rank);
        for (int d = rank - 1; d >= 0; --d) {
            source[d] = p.input_shape(input, d);
            require(source[d] >= 0 && count <= size_t(INT32_MAX) && p.input_stride(input, d) == int(count),
                    "Invalid low precision fused contiguous descriptor");
            require(!source[d] || count <= size_t(INT32_MAX) / size_t(source[d]), "Low precision fused dimensions exceed Int32 indexing");
            count *= size_t(source[d]);
        }
        for (int length : source) {
            require(!length || prefix <= size_t(INT32_MAX) / size_t(length), "Low precision fused prefix exceeds Int32 indexing");
            prefix *= size_t(length);
        }
        require(count == inputs[i]->count, "Low precision fused input shape differs from storage");
        std::vector<int> combined(std::max(shape.size(), source.size()), 1);
        for (size_t j = 0; j < combined.size(); ++j) {
            const int a = j < shape.size() ? shape[shape.size() - 1 - j] : 1;
            const int b = j < source.size() ? source[source.size() - 1 - j] : 1;
            require(a == b || a == 1 || b == 1, "Low precision operands cannot broadcast");
            combined[combined.size() - 1 - j] = a == 1 ? b : a;
        }
        shape = std::move(combined);
    }
    require(p.output_rank() == int(shape.size()), "Low precision fused output rank differs");
    size_t count = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        require(p.output_shape(int(d)) == shape[d], "Low precision fused output shape differs");
        require(!shape[d] || count <= size_t(INT32_MAX) / size_t(shape[d]), "Low precision output exceeds Int32 indexing");
        count *= size_t(shape[d]);
    }
    require(count == output->count && p.output_count() == int(count), "Low precision fused output differs from storage");
    if (op == LowPrecisionFused::Rotary) {
        require(!shape.empty() && shape.back() % 2 == 0 && p.input_rank(KernelInput::A) == int(shape.size()),
                "Low precision rotary input requires an even trailing dimension");
        for (size_t d = 0; d < shape.size(); ++d)
            require(p.input_shape(KernelInput::A, int(d)) == shape[d], "Rotary factors may not expand the input");
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer());
        const auto b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        if (!overlap) continue;
        require(op != LowPrecisionFused::Rotary, "Rotary output overlaps an operand");
        require(a == b && output->dtype == inputs[i]->dtype && output->count == inputs[i]->count &&
                p.output_rank() == p.input_rank(KernelInput(i)), "Unsafe overlapping low precision fused buffers");
        for (size_t d = 0; d < shape.size(); ++d)
            require(shape[d] == p.input_shape(KernelInput(i), int(d)), "Overlapping low precision broadcast operand");
    }
}
} // namespace elman
