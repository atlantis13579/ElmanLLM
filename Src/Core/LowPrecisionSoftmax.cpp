#include "Operators/low_precision_softmax.h"
#include <stdexcept>

namespace elman {
void validate_low_precision_softmax(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool value, const char* message) { if (!value) throw std::invalid_argument(message); };
    const int mode = p.attribute(1);
    require(mode >= 0 && mode <= 2, "Invalid low precision softmax mode");
    const int arity = mode == 2 ? 2 : 1;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4 && inputs[0], "Missing low precision softmax operand");
    if (arity == 2) require(bool(inputs[1]), "Missing low precision softmax upstream");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected low precision softmax operand");
    if (mode == 1) require(output->dtype == DType::Float32 && is_low_precision_float(inputs[0]->dtype), "Invalid wide softmax dtypes");
    else if (mode == 2) require(is_low_precision_float(output->dtype) && inputs[0]->dtype == DType::Float32 && inputs[1]->dtype == output->dtype, "Invalid softmax gradient dtypes");
    else require(is_low_precision_float(output->dtype) && inputs[0]->dtype == output->dtype, "Low precision softmax requires matching dtypes");
    for (int i = arity; i < 4; ++i) require(p.input_rank(KernelInput(i)) == -1, "Unexpected low precision softmax descriptor");
    const int rank = p.input_rank(KernelInput::A), axis = p.attribute(0);
    require(rank >= 0 && rank <= 8 && p.output_rank() == rank, "Invalid Low precision softmax rank");
    require(rank == 0 ? axis == 0 : axis >= 0 && axis < rank, "Invalid Low precision softmax axis");
    size_t count = 1, prefix = 1;
    for (int d = rank - 1; d >= 0; --d) {
        const int length = p.input_shape(KernelInput::A, d);
        require(length >= 0 && p.output_shape(d) == length && p.input_stride(KernelInput::A, d) == int(count),
                "Invalid Low precision softmax contiguous shape");
        require(length == 0 || count <= size_t(INT32_MAX) / size_t(length), "Low precision softmax dimensions overflow");
        count *= size_t(length);
    }
    for (int d = 0; d < rank; ++d) {
        const int length = p.input_shape(KernelInput::A, d);
        require(length == 0 || prefix <= size_t(INT32_MAX) / size_t(length), "Low precision softmax prefix dimensions overflow");
        prefix *= size_t(length);
    }
    require(count == inputs[0]->count && count == output->count && p.output_count() == int(count), "Low precision softmax element counts differ");
    if (arity == 2) {
        require(inputs[1]->count == count && p.input_rank(KernelInput::B) == rank, "Invalid softmax upstream size");
        for (int d = 0; d < rank; ++d)
            require(p.input_shape(KernelInput::B, d) == p.input_shape(KernelInput::A, d) &&
                    p.input_stride(KernelInput::B, d) == p.input_stride(KernelInput::A, d), "Invalid softmax upstream layout");
    }
    for (int i = 0; i < arity; ++i) if (output->bytes && inputs[i]->bytes) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        require(!overlap || (mode == 0 && a == b), "Overlapping low precision softmax buffers");
    }
}
} // namespace elman
