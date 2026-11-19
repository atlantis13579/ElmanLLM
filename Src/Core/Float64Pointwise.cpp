#include "Operators/float64_pointwise.h"
#include <algorithm>
#include <stdexcept>

namespace elman {
void validate_float64_pointwise(Float64Pointwise op, const BufferPtr& output,
                                const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    auto floating = [](DType type) { return is_compute_float(type) || type == DType::Float64; };
    const int arity = float64_pointwise_arity(op);
    const bool fusion = op == Float64Pointwise::SiluMul || op == Float64Pointwise::Rotary;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid Float64 pointwise operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected Float64 pointwise operand");
    if (fusion) for (int i = arity; i < 4; ++i)
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected Double fused operand descriptor");
    if (op == Float64Pointwise::LeakyRelu)
        require(p.attribute(2) == 0 || p.attribute(2) == 1, "Invalid Double leaky ReLU coefficient encoding");
    std::vector<int> shape;
    for (int i = 0; i < arity; ++i) {
        require(bool(inputs[i]), "Missing Float64 pointwise operand");
        require(!fusion || inputs[i]->dtype == DType::Float64, "Double fused operation requires matching Double operands");
        const auto input = KernelInput(i);
        const int rank = p.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid Float64 pointwise input rank");
        size_t count = 1, prefix = 1;
        std::vector<int> source(rank);
        for (int d = rank - 1; d >= 0; --d) {
            source[d] = p.input_shape(input, d);
            require(source[d] >= 0 && count <= size_t(INT32_MAX) && p.input_stride(input, d) == int(count),
                    "Invalid Float64 pointwise contiguous descriptor");
            require(source[d] == 0 || count <= size_t(INT32_MAX) / size_t(source[d]), "Float64 pointwise dimensions exceed Int32 indexing");
            count *= size_t(source[d]);
        }
        for (int length : source) {
            require(length == 0 || prefix <= size_t(INT32_MAX) / size_t(length), "Float64 pointwise prefix dimensions exceed Int32 indexing");
            prefix *= size_t(length);
        }
        require(count == inputs[i]->count, "Float64 pointwise input shape differs from storage");
        std::vector<int> combined(std::max(shape.size(), source.size()), 1);
        for (size_t j = 0; j < combined.size(); ++j) {
            const int x = j < shape.size() ? shape[shape.size() - 1 - j] : 1;
            const int y = j < source.size() ? source[source.size() - 1 - j] : 1;
            require(x == y || x == 1 || y == 1, "Float64 pointwise inputs cannot broadcast");
            combined[combined.size() - 1 - j] = x == 1 ? y : x;
        }
        shape = std::move(combined);
        require(op == Float64Pointwise::Where && i == 0 ? inputs[i]->dtype == DType::Bool : floating(inputs[i]->dtype),
                "Invalid Float64 pointwise input dtype");
    }
    require(p.output_rank() == int(shape.size()), "Invalid Float64 pointwise output rank");
    size_t count = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        require(p.output_shape(int(d)) == shape[d], "Invalid Float64 pointwise output shape");
        require(shape[d] == 0 || count <= size_t(INT32_MAX) / size_t(shape[d]), "Float64 pointwise output exceeds Int32 indexing");
        count *= size_t(shape[d]);
    }
    require(count == output->count && p.output_count() == int(count), "Float64 pointwise output shape differs from storage");
    if (op == Float64Pointwise::Rotary) {
        require(!shape.empty() && shape.back() % 2 == 0 && p.input_rank(KernelInput::A) == int(shape.size()),
                "Double rotary input requires an even trailing dimension");
        for (size_t d = 0; d < shape.size(); ++d)
            require(p.input_shape(KernelInput::A, int(d)) == shape[d], "Double rotary factors may not expand the input");
    }
    const bool boolean = op >= Float64Pointwise::Eq && op <= Float64Pointwise::LogicalNot;
    require(op == Float64Pointwise::Cast ? floating(output->dtype) : output->dtype == (boolean ? DType::Bool : DType::Float64),
            "Invalid Float64 pointwise output dtype");
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto first = reinterpret_cast<uintptr_t>(output->raw_pointer());
        const auto second = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = first <= second ? second - first < output->bytes : first - second < inputs[i]->bytes;
        if (!overlap) continue;
        require(op != Float64Pointwise::Rotary, "Double rotary output overlaps an operand");
        require(first == second && output->dtype == inputs[i]->dtype && output->count == inputs[i]->count &&
                p.output_rank() == p.input_rank(KernelInput(i)), "Unsafe overlapping Float64 pointwise buffers");
        for (size_t d = 0; d < shape.size(); ++d)
            require(shape[d] == p.input_shape(KernelInput(i), int(d)), "Overlapping Float64 broadcast operand");
    }
}
} // namespace elman
