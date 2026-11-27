#include "Kernals/low_precision_random.h"
#include "Kernals/float64_random_math.h"
#include "Kernals/attributes.h"
#include <limits>
#include <stdexcept>

namespace elman {
void validate_low_precision_random(LowPrecisionRandom op, const BufferPtr& output,
        const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool dropout = op == LowPrecisionRandom::Dropout;
    const int arity = dropout ? 1 : 0;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid low precision random operand count");
    for (int i = arity; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected low precision random operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected low precision random operand descriptor");
    }
    require(is_low_precision_float(output->dtype), "Low precision random operation requires a low precision output");
    require(p.attribute(8) == 0 || p.attribute(8) == 1, "Invalid low precision random parameter encoding");
    require(p.attribute(kernal_attr::dropout::kBackward) == 0 || (dropout && p.attribute(kernal_attr::dropout::kBackward) == 1), "Invalid low precision random derivative mode");
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8, "Invalid low precision random rank");
    if (dropout) require(inputs[0] && inputs[0]->dtype == output->dtype && p.input_rank(KernelInput::A) == rank,
                         "Low precision dropout requires matching operands");
    size_t forward = 1, reverse = 1;
    for (int d = 0; d < rank; ++d) {
        const int length = p.output_shape(d), backward = p.output_shape(rank - d - 1);
        require(length >= 0 && backward >= 0 &&
            (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
            (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "Low precision random dimensions overflow");
        if (dropout) require(p.input_shape(KernelInput::A, d) == length &&
            p.input_stride(KernelInput::A, rank - d - 1) == int(reverse), "Invalid low precision dropout contiguous shape");
        forward *= size_t(length); reverse *= size_t(backward);
    }
    require(forward == output->count && p.output_count() == int(forward) && (!dropout || inputs[0]->count == forward), "Low precision random shape differs from storage");
    const double first = float64_random_parameter(p, 0), second = float64_random_parameter(p, 1);
    if (op == LowPrecisionRandom::Uniform) {
        const double maximum = std::numeric_limits<float>::max();
        require(std::isfinite(first) && std::isfinite(second) && first >= -maximum && second <= maximum && first <= second && second - first <= maximum,
                "Invalid low precision uniform range");
    } else if (op == LowPrecisionRandom::Normal) require(second >= 0., "Invalid low precision normal standard deviation");
    else {
        require(first >= 0. && first <= 1. && (p.attribute(3) == 0 || p.attribute(3) == 1), "Invalid low precision dropout parameters");
        if (output->bytes && inputs[0]->bytes) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
            const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[0]->bytes;
            require(!overlap || a == b, "Partially overlapping low precision dropout buffers");
        }
    }
}
} // namespace elman
