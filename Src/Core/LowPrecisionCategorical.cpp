#include "Kernals/low_precision_categorical.h"
#include <stdexcept>

namespace elman {
void validate_low_precision_categorical(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(inputs.size() >= 3 && inputs.size() <= 4 && inputs[0] && inputs[1] && inputs[2] &&
        (inputs.size() == 3 || !inputs[3]) && p.input_rank(KernelInput::D) == -1, "Low precision categorical requires three operands");
    require(is_low_precision_float(inputs[0]->dtype), "Low precision categorical requires low precision weights");
    require(output->dtype == DType::Int32 && p.output_rank() == 1, "Low precision categorical requires an Int32 vector output");
    const int rows = p.output_shape(0), width = p.input_shape(KernelInput::A, 1);
    require(rows >= 0 && width > 0 && size_t(rows) <= size_t(INT32_MAX) / size_t(width) &&
        p.output_count() == rows && output->count == size_t(rows), "Invalid low precision categorical dimensions");
    for (int i = 0; i < 3; ++i) {
        const KernelInput input = KernelInput(i);
        require(inputs[i]->dtype == inputs[0]->dtype || (i != 0 && inputs[i]->dtype == DType::Float32),
            "Low precision cumulative probabilities and uniforms require matching low storage or explicit Float32");
        require(p.input_rank(input) == (i == 2 ? 1 : 2) && p.input_shape(input, 0) == rows &&
            p.input_stride(input, 0) == (i == 2 ? 1 : width) &&
            (i == 2 || (p.input_shape(input, 1) == width && p.input_stride(input, 1) == 1)) &&
            inputs[i]->count == size_t(rows) * size_t(i == 2 ? 1 : width), "Invalid low precision categorical operand descriptor");
        if (output->bytes && inputs[i]->bytes) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
            require(!(a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes), "Overlapping low precision categorical buffers");
        }
    }
    for (int i = 0; i < KernelParams::kAttributeCount; ++i) require(p.attribute(i) == 0, "Unexpected low precision categorical attribute");
}
} // namespace elman
