#include "Kernals/float64_normalization_math.h"
#include "Core/LocalFunction.h"
#include <stdexcept>

namespace elman {
void validate_float64_normalization(Float64Normalization op, const BufferPtr& output,
        const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool layer = op == Float64Normalization::Layer || op == Float64Normalization::Rms;
    const bool batch = op == Float64Normalization::Batch;
    const bool dx = op == Float64Normalization::GroupInputGradient, dw = op == Float64Normalization::GroupWeightGradient;
    require(inputs.size() >= 1 && inputs.size() <= 4 && inputs[0], "Double normalization requires an input");
    LocalFunction present{[&](int i) { return size_t(i) < inputs.size() && bool(inputs[i]); }};
    LocalFunction shape{[&](int input) {
        const bool out = input < 0;
        const KernelInput id = KernelInput(out ? 0 : input);
        const BufferPtr buffer = out ? output : inputs[input];
        const int rank = out ? p.output_rank() : p.input_rank(id);
        require(rank >= 0 && rank <= 8, "Invalid Double normalization rank");
        require(buffer->dtype == DType::Float64, "Double normalization requires matching Double operands");
        std::vector<int> result;
        size_t count = 1, reverse = 1;
        for (int d = 0; d < rank; ++d) {
            const int length = out ? p.output_shape(d) : p.input_shape(id, d);
            const int backward = out ? p.output_shape(rank - d - 1) : p.input_shape(id, rank - d - 1);
            require(length >= 0 && backward >= 0 && (length == 0 || count <= size_t(INT32_MAX) / size_t(length)) &&
                (backward == 0 || reverse <= size_t(INT32_MAX) / size_t(backward)), "Double normalization dimensions overflow");
            if (!out) require(p.input_stride(id, rank - d - 1) == int(reverse), "Invalid Double normalization strides");
            result.push_back(length); count *= size_t(length); reverse *= size_t(backward);
        }
        require(count == buffer->count && (!out || p.output_count() == int(count)), "Double normalization shape differs from storage");
        return result;
    }};
    const std::vector<int> output_shape = shape(-1), input_shape = shape(0);
    for (int i = 0; i < 4; ++i) {
        const bool needed = i == 0 || ((batch || dx) && i < 3) || (dw && i == 1);
        require(!needed || present(i), "Missing Double normalization operand");
        if (!present(i)) require(p.input_rank(KernelInput(i)) == -1, "Unexpected Double normalization operand descriptor");
        else {
            require(i < 3 && (!dw || i < 2), "Unexpected Double normalization operand");
            shape(i);
            if (output->bytes && inputs[i]->bytes) {
                const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
                require(!(a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes), "Overlapping Double normalization buffers");
            }
        }
    }
    require(p.attribute(7) == 0 || p.attribute(7) == 1, "Invalid Double normalization epsilon encoding");
    if (!layer && !batch) require(p.attribute(2) == 0 || p.attribute(2) == 1, "Invalid Double group normalization epsilon encoding");
    require(float64_normalization_epsilon(op, p) >= 0., "Invalid Double normalization epsilon");
    if (op == Float64Normalization::Rms) require(p.attribute(2) == 0 || p.attribute(2) == 1, "Invalid Double RMS normalization rounding");
    std::vector<int> affine;
    if (layer) {
        const int trailing = p.attribute(0);
        require(trailing >= 1 && trailing <= int(input_shape.size()), "Invalid Double normalization trailing dimensions");
        affine = std::vector<int>(input_shape.end() - trailing, input_shape.end());
    } else {
        require(input_shape.size() >= 2, "Double channel normalization requires rank two or greater");
        const int channels = input_shape[1];
        affine = {channels};
        if (!batch) {
            const int groups = p.attribute(0);
            require(groups > 0 && channels % groups == 0, "Invalid Double group normalization groups");
            require(output->backend != Backend::CUDA || channels > 0 || (op == Float64Normalization::Group && !present(1) && !present(2)), "CUDA Double group normalization requires nonzero channels");
            require(inputs[0]->count / size_t(groups) != 1, "Double group normalization requires multiple values across the batch");
        }
    }
    require(output_shape == (dw ? affine : input_shape), "Invalid Double normalization output shape");
    if (present(1)) require(shape(1) == (dw ? input_shape : affine), "Invalid Double normalization second operand shape");
    if (present(2)) require(shape(2) == (dx ? input_shape : affine), "Invalid Double normalization third operand shape");
}
} // namespace elman
