#include "Operators/affine_quantization.h"
#include <limits>
#include <stdexcept>

namespace elman {
void validate_affine_quantization(AffineQuantization operation, const BufferPtr& output,
                                  const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(operation != AffineQuantization::None, "Invalid affine quantization operation");
    const int arity = operation == AffineQuantization::Embedding ? 4 : 3;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid affine quantization operand count");
    for (int i = 0; i < 4; ++i) {
        if (i < arity) require(inputs[i] && inputs[i]->backend == output->backend, "Missing or mixed-backend quantization operand");
        else require((size_t(i) >= inputs.size() || !inputs[i]) && p.input_rank(KernelInput(i)) == -1,
                     "Unexpected affine quantization operand");
    }
    const auto code_dtype = operation == AffineQuantization::Quantize ? output->dtype : inputs[0]->dtype;
    const auto float_dtype = operation == AffineQuantization::Quantize ? inputs[0]->dtype : output->dtype;
    require(quantized_storage_bits(code_dtype) && (operation == AffineQuantization::Relu ? output->dtype == code_dtype :
                is_arithmetic_float(float_dtype)), "Invalid affine conversion dtype pair");
    require(is_arithmetic_float(inputs[1]->dtype), "Quantization scales require real floating storage");
    require(inputs[2]->dtype == DType::Int32 || inputs[2]->dtype == DType::Int64, "Invalid quantization zero-point dtype");
    require(p.output_dtype() == int(output->dtype) && p.output_count() >= 0 && size_t(p.output_count()) == output->count,
            "Affine quantization output descriptor mismatch");

    auto shape = [&](const BufferPtr& buffer, int rank, int slot, bool destination) {
        require(rank >= 0 && rank <= 8, "Invalid affine quantization tensor rank");
        size_t prefix = 1, suffix = 1;
        const size_t limit = size_t(std::numeric_limits<int>::max());
        for (int d = 0; d < rank; ++d) {
            const int length = p.v[slot + d], reverse = p.v[slot + rank - d - 1];
            require(length >= 0 && reverse >= 0, "Negative affine quantization dimension");
            require((!length || prefix <= limit / size_t(length)) && (!reverse || suffix <= limit / size_t(reverse)),
                    "Affine quantization tensor exceeds Int32 indexing");
            if (!destination)
                require(suffix <= limit && p.v[slot + 8 + rank - d - 1] == int(suffix), "Invalid affine quantization contiguous stride");
            prefix *= size_t(length);
            suffix *= size_t(reverse);
        }
        require(prefix <= limit && prefix == buffer->count, "Affine quantization shape differs from storage");
    };
    shape(output, p.output_rank(), KernelParams::kOutputShapeSlot, true);
    for (int i = 0; i < arity; ++i) {
        const auto input = KernelInput(i);
        require(p.input_dtype(input) == int(inputs[i]->dtype), "Affine quantization input dtype descriptor mismatch");
        shape(inputs[i], p.input_rank(input), KernelParams::input_slot(input) + KernelParams::kInputShapeOffset, false);
        if (output->bytes && inputs[i]->bytes) {
            const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
            require(a <= b ? b - a >= output->bytes : a - b >= inputs[i]->bytes,
                    "Affine quantization output overlaps an operand");
        }
    }
    const int group_size = p.attribute(0), rank = p.input_rank(KernelInput::A);
    require(group_size > 0, "Affine quantization group size must be positive");
    const int columns = rank ? p.input_shape(KernelInput::A, rank - 1) : 1;
    int64_t rows = 1;
    for (int d = 0; d + 1 < rank; ++d) rows *= p.input_shape(KernelInput::A, d);
    require(rows <= std::numeric_limits<int>::max(), "Affine quantization row count exceeds Int32 indexing");
    const int groups = int((int64_t(columns) + group_size - 1) / group_size);
    for (auto input : {KernelInput::B, KernelInput::C}) {
        const int parameter_rank = p.input_rank(input), lengths[2] = {int(rows), groups};
        require(parameter_rank <= 2, "Affine quantization metadata rank exceeds two");
        for (int d = 0; d < parameter_rank; ++d)
            require(p.input_shape(input, d) == 1 || p.input_shape(input, d) == lengths[d + 2 - parameter_rank],
                    "Affine quantization metadata does not broadcast to groups");
    }
    if (operation == AffineQuantization::Embedding) {
        require(rank == 2 && (inputs[3]->dtype == DType::Int32 || inputs[3]->dtype == DType::Int64),
                "Quantized embedding requires a matrix and integer indices");
        const int index_rank = p.input_rank(KernelInput::D);
        require(index_rank < 8 && p.output_rank() == index_rank + 1 && p.output_shape(index_rank) == columns,
                "Quantized embedding output rank or width differs");
        for (int d = 0; d < index_rank; ++d)
            require(p.output_shape(d) == p.input_shape(KernelInput::D, d), "Quantized embedding output index shape differs");
    } else {
        require(p.output_rank() == rank, "Affine conversion cannot change rank");
        for (int d = 0; d < rank; ++d)
            require(p.output_shape(d) == p.input_shape(KernelInput::A, d), "Affine conversion cannot change shape");
    }
}
} // namespace elman
