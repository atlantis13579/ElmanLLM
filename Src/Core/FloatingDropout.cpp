#include "Kernals/floating_dropout.h"
#include "Kernals/float64_random_math.h"
#include "Kernals/attributes.h"
#include <stdexcept>

namespace elman {
void validate_floating_dropout(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(inputs.size() >= 1 && inputs.size() <= 4 && inputs[0] && inputs[0]->dtype == output->dtype && is_compute_float(output->dtype), "Floating dropout requires matching operands");
    for (int i = 1; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected floating dropout operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected floating dropout descriptor");
    }
    require(p.attribute(8) == 0 || p.attribute(8) == 1, "Invalid floating dropout parameter encoding");
    require(p.attribute(operator_attr::dropout::kBackward) == 0 || p.attribute(operator_attr::dropout::kBackward) == 1, "Invalid floating dropout derivative mode");
    const double probability = float64_random_parameter(p, 0);
    require(probability >= 0. && probability <= 1. && (p.attribute(3) == 0 || p.attribute(3) == 1), "Invalid floating dropout probability or training flag");
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8 && p.input_rank(KernelInput::A) == rank, "Invalid floating dropout rank");
    size_t forward = 1, reverse = 1;
    for (int d = 0; d < rank; ++d) {
        const int length = p.output_shape(d), backward = p.output_shape(rank - d - 1);
        require(length >= 0 && backward >= 0 &&
            (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
            (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "Floating dropout dimensions overflow");
        require(p.input_shape(KernelInput::A, d) == length && p.input_stride(KernelInput::A, rank - d - 1) == int(reverse), "Invalid floating dropout contiguous shape");
        forward *= size_t(length); reverse *= size_t(backward);
    }
    require(forward == output->count && forward == inputs[0]->count && p.output_count() == int(forward), "Floating dropout shape differs from storage");
    if (output->bytes && inputs[0]->bytes) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[0]->bytes;
        require(!overlap || a == b, "Partially overlapping floating dropout buffers");
    }
}
} // namespace elman
