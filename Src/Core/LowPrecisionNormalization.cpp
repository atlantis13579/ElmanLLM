#include "Operators/low_precision_normalization.h"
#include <cstring>
#include <stdexcept>
namespace elman {
void validate_low_precision_normalization(LowPrecisionNormalization op, const BufferPtr& output,
                                          const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool value, const char* message) { if (!value) throw std::invalid_argument(message); };
    const int mode = low_precision_normalization_mode(op, p);
    const bool named_gradient = low_precision_normalization_base(op) != op;
    op = low_precision_normalization_base(op);
    const bool trailing = op == LowPrecisionNormalization::Layer || op == LowPrecisionNormalization::Rms;
    const bool rms = op == LowPrecisionNormalization::Rms, group = op == LowPrecisionNormalization::Group;
    const bool batch = op == LowPrecisionNormalization::Batch, training = batch && p.attribute(9) == 1;
    require(mode >= 0 && mode <= 3 && !(rms && mode == 3), "Invalid low precision normalization derivative");
    require(!named_gradient || p.attribute(8) == 0, "Conflicting named normalization derivative");
    require(p.attribute(7) == 0 || p.attribute(7) == 1, "Invalid normalization epsilon encoding");
    require(p.attribute(2) == 0 || ((rms || group) && p.attribute(2) == 1), "Unsupported normalization rounding or epsilon encoding");
    require(p.attribute(9) == 0 || (batch && p.attribute(9) == 1), "Invalid normalization training mode");
    require(!batch || training || mode == 0, "Inference normalization has no training derivative");
    require(!inputs.empty() && inputs.size() <= 4 && inputs[0], "Missing normalization operand");
    const bool native_float = op == LowPrecisionNormalization::Layer && p.attribute(10) == 1 &&
        output->backend == Backend::CPU && output->dtype == DType::Float32;
    require(is_low_precision_float(output->dtype) || native_float, "Normalization requires a low precision output or private CPU Float32 LayerNorm");
    require(p.attribute(10) == int(native_float), "Invalid native normalization mode");
    auto present = [&](int i) { return size_t(i) < inputs.size() && bool(inputs[i]); };
    for (int i = 0; i < 4; ++i) {
        const bool needed = i == 0 || (((batch && mode == 0) || mode == 1) && i < 3) || (mode >= 2 && i == 1);
        require(!needed || present(i), "Missing normalization operand");
        if (!present(i)) {
            require(p.input_rank(KernelInput(i)) == -1, "Unexpected normalization descriptor");
            continue;
        }
        require(i < 3 && (mode < 2 || i < 2), "Unexpected normalization operand");
        require(inputs[i]->dtype == output->dtype, "Normalization dtypes differ");
        const auto input = KernelInput(i); const int rank = p.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid normalization input rank");
        size_t count = 1, prefix = 1;
        for (int d = rank - 1; d >= 0; --d) {
            int n = p.input_shape(input, d);
            require(n >= 0 && p.input_stride(input, d) == int(count), "Invalid normalization input layout");
            require(n == 0 || count <= size_t(INT32_MAX) / size_t(n), "Normalization dimensions overflow"); count *= size_t(n);
        }
        for (int d = 0; d < rank; ++d) {
            int n = p.input_shape(input, d);
            require(n == 0 || prefix <= size_t(INT32_MAX) / size_t(n), "Normalization prefix dimensions overflow"); prefix *= size_t(n);
        }
        require(count == inputs[i]->count, "Normalization input count differs");
        if (output->bytes && inputs[i]->bytes) {
            uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
            require(!(a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes), "Overlapping normalization buffers");
        }
    }
    const int rank = p.input_rank(KernelInput::A), dimensions = trailing ? p.attribute(0) : 1;
    require(trailing ? dimensions > 0 && dimensions <= rank : rank >= 2, "Invalid normalization dimensions");
    require(low_precision_normalization_epsilon(op, p) >= 0., "Invalid normalization epsilon");
    if (training) {
        const int channels = p.input_shape(KernelInput::A, 1);
        require(channels > 0 && inputs[0]->count / channels != 1, "Invalid training batch normalization channels");
    }
    if (group) {
        const int groups = p.attribute(0), channels = p.input_shape(KernelInput::A, 1);
        require(groups > 0 && channels % groups == 0 && inputs[0]->count / groups != 1, "Invalid normalization groups");
        require(p.input_shape(KernelInput::A, 0) <= INT32_MAX / groups, "Normalization group rows overflow");
        require(output->backend == Backend::CPU || channels > 0 || (mode == 0 && !present(1) && !present(2)), "CUDA normalization requires nonzero channels with affine parameters");
    }
    const int expected_rank = mode < 2 ? rank : dimensions;
    require(p.output_rank() == expected_rank, "Invalid normalization output rank");
    size_t output_count = 1;
    for (int d = 0; d < expected_rank; ++d) {
        const int n = mode < 2 ? p.input_shape(KernelInput::A, d) : trailing ? p.input_shape(KernelInput::A, rank - dimensions + d) : p.input_shape(KernelInput::A, 1);
        require(p.output_shape(d) == n, "Invalid normalization output shape"); output_count *= size_t(n);
    }
    require(output_count == output->count && p.output_count() == int(output_count), "Invalid normalization output count");
    for (int i = 1; i < 3; ++i) {
        if (!present(i)) continue;
        const auto input = KernelInput(i);
        const bool gradient = mode >= 2 || (mode == 1 && i == 2);
        require(p.input_rank(input) == (gradient ? rank : dimensions), "Invalid normalization parameter rank");
        for (int d = 0; d < p.input_rank(input); ++d) {
            const int n = gradient ? p.input_shape(KernelInput::A, d) : trailing ? p.input_shape(KernelInput::A, rank - dimensions + d) : p.input_shape(KernelInput::A, 1);
            require(p.input_shape(input, d) == n, "Invalid normalization parameter shape");
        }
    }
}
} // namespace elman
