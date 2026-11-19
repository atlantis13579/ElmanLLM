#include "Operators/low_precision_cast.h"
#include <stdexcept>

namespace elman {
void validate_low_precision_cast(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    auto floating = [](DType dtype) { return is_arithmetic_float(dtype) || is_low_precision_float(dtype); };
    require(!inputs.empty() && inputs.size() <= 4 && inputs[0], "Low precision cast requires one input");
    for (size_t i = 1; i < inputs.size(); ++i) require(!inputs[i], "Unexpected low precision cast input");
    for (int i = 1; i < 4; ++i) require(p.input_rank(KernelInput(i)) == -1, "Unexpected low precision cast descriptor");
    require(floating(output->dtype) && floating(inputs[0]->dtype), "Low precision cast requires floating dtypes");
    const int rank = p.input_rank(KernelInput::A);
    require(rank >= 0 && rank <= 8 && p.output_rank() == rank, "Invalid low precision cast rank");
    size_t count = 1, prefix = 1;
    for (int d = rank - 1; d >= 0; --d) {
        const int length = p.input_shape(KernelInput::A, d);
        const int front = p.input_shape(KernelInput::A, rank - d - 1);
        require(length >= 0 && front >= 0 && p.output_shape(d) == length, "Invalid low precision cast shape");
        require(count <= size_t(INT32_MAX) && p.input_stride(KernelInput::A, d) == int(count), "Invalid low precision cast stride");
        require((!length || count <= size_t(INT32_MAX) / size_t(length)) &&
                (!front || prefix <= size_t(INT32_MAX) / size_t(front)), "Low precision cast shape exceeds Int32 indexing");
        count *= size_t(length);
        prefix *= size_t(front);
    }
    require(count == output->count && count == inputs[0]->count && p.output_count() == int(count),
            "Low precision cast descriptor differs from storage");
    if (output->bytes && inputs[0]->bytes) {
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer());
        const auto b = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[0]->bytes;
        require(!overlap || (a == b && output->dtype == inputs[0]->dtype), "Unsafe overlapping low precision cast buffers");
    }
}
} // namespace elman
