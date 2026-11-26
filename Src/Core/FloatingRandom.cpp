#include "Kernals/floating_random.h"
#include "Kernals/floating_random_math.h"
#include <limits>
#include <stdexcept>

namespace elman {
void validate_floating_random(bool normal, const BufferPtr& output,
        const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(is_compute_float(output->dtype) && inputs.size() <= 4, "Floating random operation requires a native floating output");
    for (int i = 0; i < 4; ++i) {
        require(size_t(i) >= inputs.size() || !inputs[i], "Unexpected floating random operand");
        require(p.input_rank(KernelInput(i)) == -1, "Unexpected floating random descriptor");
    }
    require((p.attribute(8) == 0 || p.attribute(8) == 1) && p.attribute(9) == 0, "Invalid floating random parameter encoding");
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8, "Invalid floating random rank");
    size_t forward = 1, reverse = 1;
    for (int d = 0; d < rank; ++d) {
        const int length = p.output_shape(d), backward = p.output_shape(rank - d - 1);
        require(length >= 0 && backward >= 0 &&
            (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
            (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "Floating random dimensions overflow");
        forward *= size_t(length); reverse *= size_t(backward);
    }
    require(forward == output->count && p.output_count() == int(forward), "Floating random shape differs from storage");
    const double first = float64_random_parameter(p, 0), second = float64_random_parameter(p, 1);
    if (normal) require(second >= 0., "Invalid floating normal standard deviation");
    else {
        const double maximum = output->dtype == DType::Float16 ? 65504. : output->dtype == DType::BFloat16 ?
            double(numeric::bfloat_to_float(0x7f7f)) : double(std::numeric_limits<float>::max());
        require(first >= -maximum && first <= maximum && second >= -maximum && second <= maximum &&
            first <= second && second - first <= maximum, "Invalid floating uniform range");
        if (output->backend == Backend::CPU) {
            const float low = floating_random_round(float(first), output->dtype), high = floating_random_round(float(second), output->dtype);
            require(double(floating_random_round(high - low, output->dtype)) <= maximum, "Rounded floating uniform range overflows");
        }
    }
}
} // namespace elman
