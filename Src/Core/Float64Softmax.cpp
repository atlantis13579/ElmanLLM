#include "Operators/float64_softmax.h"
#include <stdexcept>

namespace elman {
void validate_float64_softmax(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool value, const char* message) { if (!value) throw std::invalid_argument(message); };
    require(inputs.size() >= 1 && inputs.size() <= 4 && inputs[0], "Double softmax requires one input");
    for (size_t i = 1; i < inputs.size(); ++i) require(!inputs[i], "Unexpected Double softmax operand");
    require(output->dtype == DType::Float64 && inputs[0]->dtype == DType::Float64, "Double softmax requires matching dtypes");
    const int rank = p.input_rank(KernelInput::A), axis = p.attribute(0);
    require(rank >= 0 && rank <= 8 && p.output_rank() == rank, "Invalid Double softmax rank");
    require(rank == 0 ? axis == 0 : axis >= 0 && axis < rank, "Invalid Double softmax axis");
    size_t count = 1, prefix = 1;
    for (int d = rank - 1; d >= 0; --d) {
        const int length = p.input_shape(KernelInput::A, d);
        require(length >= 0 && p.output_shape(d) == length && p.input_stride(KernelInput::A, d) == int(count),
                "Invalid Double softmax contiguous shape");
        require(length == 0 || count <= size_t(INT32_MAX) / size_t(length), "Double softmax dimensions overflow");
        count *= size_t(length);
    }
    for (int d = 0; d < rank; ++d) {
        const int length = p.input_shape(KernelInput::A, d);
        require(length == 0 || prefix <= size_t(INT32_MAX) / size_t(length), "Double softmax prefix dimensions overflow");
        prefix *= size_t(length);
    }
    require(count == inputs[0]->count && count == output->count && p.output_count() == int(count), "Double softmax element counts differ");
    if (output->bytes && inputs[0]->bytes) {
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[0]->bytes;
        require(!overlap || a == b, "Partially overlapping Double softmax buffers");
    }
}
} // namespace elman
