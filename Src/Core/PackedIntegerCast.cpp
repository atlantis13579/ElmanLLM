#include "Operators/packed_integer_cast.h"
#include <limits>
#include <stdexcept>

namespace elman {
void validate_packed_integer_cast(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(inputs.size() >= 1 && inputs.size() <= 4 && inputs[0], "Packed integer cast requires one operand");
    bool (*const numeric)(DType) = [](DType dtype) { return is_arithmetic_integer(dtype) || is_arithmetic_float(dtype) || is_low_precision_float(dtype); };
    require(numeric(output->dtype) && numeric(inputs[0]->dtype), "Packed integer cast requires numeric operands");
    require(output->backend == inputs[0]->backend && p.output_dtype() == int(output->dtype) &&
            p.input_dtype(KernelInput::A) == int(inputs[0]->dtype), "Packed integer cast backend or dtype mismatch");
    for (size_t i = 1; i < inputs.size(); ++i) require(!inputs[i], "Unexpected packed integer cast operand");
    for (int i = 1; i < 4; ++i) require(p.input_rank(KernelInput(i)) == -1, "Unexpected packed integer cast descriptor");
    const int rank = p.input_rank(KernelInput::A);
    require(rank >= 0 && rank <= 8 && p.output_rank() == rank, "Invalid packed integer cast rank");
    size_t count = 1, prefix = 1;
    const size_t limit = size_t(std::numeric_limits<int>::max());
    for (int d = rank - 1; d >= 0; --d) {
        const int length = p.input_shape(KernelInput::A, d), front = p.input_shape(KernelInput::A, rank - d - 1);
        require(length >= 0 && front >= 0 && p.output_shape(d) == length, "Invalid packed integer cast shape");
        require(count <= limit && p.input_stride(KernelInput::A, d) == int(count), "Invalid packed integer cast stride");
        require((!length || count <= limit / size_t(length)) && (!front || prefix <= limit / size_t(front)),
                "Packed integer cast dimensions exceed Int32 indexing");
        count *= size_t(length);
        prefix *= size_t(front);
    }
    require(count <= limit && count == output->count && count == inputs[0]->count && p.output_count() == int(count),
            "Packed integer cast descriptor differs from storage");
    if (output->bytes && inputs[0]->bytes) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[0]->bytes;
        require(!overlap || (a == b && output->dtype == inputs[0]->dtype), "Unsafe overlapping packed integer cast buffers");
    }
}
} // namespace elman
