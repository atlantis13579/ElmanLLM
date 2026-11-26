#include "Kernals/float64_random_math.h"
#include <stdexcept>

namespace elman {
void validate_float64_random(Float64Random op, const BufferPtr& output,
        const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool dropout = op == Float64Random::Dropout;
    const int arity = dropout ? 1 : 0;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid Double random operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected Double random operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected Double random operand descriptor");
    }
    require(output->dtype == DType::Float64, "Double random operation requires a Double output");
    require(p.attribute(8) == 0 || p.attribute(8) == 1, "Invalid Double random parameter encoding");
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8, "Invalid Double random rank");
    if (dropout) require(inputs[0] && inputs[0]->dtype == DType::Float64 && p.input_rank(KernelInput::A) == rank,
                         "Double dropout requires a matching Double input");
    size_t forward = 1, reverse = 1;
    for (int d = 0; d < rank; ++d) {
        const int length = p.output_shape(d), backward = p.output_shape(rank - d - 1);
        require(length >= 0 && backward >= 0 &&
                (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
                (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "Double random dimensions overflow");
        if (dropout) require(p.input_shape(KernelInput::A, d) == length &&
            p.input_stride(KernelInput::A, rank - d - 1) == int(reverse), "Invalid Double dropout contiguous shape");
        forward *= size_t(length); reverse *= size_t(backward);
    }
    require(forward == output->count && p.output_count() == int(forward) && (!dropout || inputs[0]->count == forward),
            "Double random shape differs from storage");
    const double first = float64_random_parameter(p, 0), second = float64_random_parameter(p, 1);
    if (op == Float64Random::Uniform)
        require(std::isfinite(first) && std::isfinite(second) && first <= second && std::isfinite(second - first), "Invalid Double uniform range");
    else if (op == Float64Random::Normal) require(second >= 0., "Invalid Double normal standard deviation");
    else {
        require(first >= 0. && first <= 1. && (p.attribute(3) == 0 || p.attribute(3) == 1), "Invalid Double dropout parameters");
        if (output->bytes && inputs[0]->bytes) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
            const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[0]->bytes;
            require(!overlap || a == b, "Partially overlapping Double dropout buffers");
        }
    }
}
} // namespace elman
