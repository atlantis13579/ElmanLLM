#include "Operators/float64_indexing.h"
#include "Core/LocalFunction.h"
#include "Core/Memory.h"
#include <algorithm>
#include <optional>
#include <cstring>
#include <stdexcept>

namespace elman {
void validate_float64_indexing(Float64Indexing op, const BufferPtr& output,
                               const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool repeat = op == Float64Indexing::RepeatGradient;
    const bool scatter = op == Float64Indexing::Scatter || op == Float64Indexing::ScatterAdd;
    const bool source_gradient = op == Float64Indexing::ScatterGradient;
    const bool selection = op == Float64Indexing::SelectionGradient;
    const bool sorting = op == Float64Indexing::SortGradient;
    const int arity = repeat ? 1 : scatter ? 3 : 2;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid Double indexing operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected Double indexing operand");
    require(output->dtype == DType::Float64, "Double indexing requires a Double output");
    LocalFunction shape{[&](std::optional<KernelInput> input, size_t expected) {
        const int rank = input ? p.input_rank(*input) : p.output_rank();
        require(rank >= 0 && rank <= 8, "Invalid Double indexing rank");
        std::vector<int> result(rank);
        size_t forward = 1, backward = 1;
        for (int d = 0; d < rank; ++d) {
            const int reverse_axis = rank - d - 1;
            const int length = input ? p.input_shape(*input, d) : p.output_shape(d);
            const int reverse = input ? p.input_shape(*input, reverse_axis) : p.output_shape(reverse_axis);
            require(length >= 0 && reverse >= 0 &&
                    (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
                    (reverse == 0 || backward <= size_t(INT32_MAX) / size_t(reverse)), "Double indexing dimensions overflow");
            require(!input || p.input_stride(*input, reverse_axis) == int(backward), "Invalid Double indexing contiguous stride");
            forward *= size_t(length); backward *= size_t(reverse); result[d] = length;
        }
        require(forward == expected, "Double indexing shape differs from storage");
        return result;
    }};
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == (i == 1 ? DType::Int32 : DType::Float64), "Invalid Double indexing operand dtype");
        const KernelInput input = KernelInput(i);
        shapes.push_back(shape(input, inputs[i]->count));
    }
    std::vector<int> target = shape(std::nullopt, output->count);
    require(p.output_count() == int(output->count), "Invalid Double indexing output count");
    const std::vector<int>& source = shapes[0];
    const int rank = int(target.size()), axis = p.attribute(0);
    int bound = 0;
    if (repeat) {
        require(source.size() >= target.size(), "Double repeat gradient removes source dimensions");
        for (int d = 0; d < rank; ++d) {
            const int from = target[d], to = source[d + source.size() - target.size()];
            require(from ? to % from == 0 : to == 0, "Invalid Double repeat gradient shape");
        }
    } else {
        require(rank >= 1 && int(source.size()) == rank && axis >= 0 && axis < rank, "Invalid Double indexing axis/rank");
        const std::vector<int>& indexed = scatter ? shapes[2] : source_gradient ? target : source;
        require(int(indexed.size()) == rank, "Double indexing operand ranks differ");
        if (selection) require(shapes[1].size() == 1 && indexed[axis] == int(inputs[1]->count), "Invalid Double selection indices");
        else require(shapes[1] == indexed, "Double indexing indices must match the selected values");
        if (scatter) require(source == target, "Double scatter destination shape differs");
        if (sorting) require(source == target, "Double sort gradient shapes differ");
        const std::vector<int>& destination = source_gradient ? source : target;
        for (int d = 0; d < rank; ++d) if (d != axis)
            require(selection || sorting ? indexed[d] == destination[d] : indexed[d] <= destination[d], "Double indexing non-axis dimensions differ");
        bound = destination[axis];
        if (inputs[1]->count) {
            MemoryReservation host(Backend::CPU, checked_memory_add(checked_memory_mul(inputs[1]->bytes, 2), sorting ? size_t(bound) : 0));
            const std::vector<uint8_t> raw = backend_download_raw(inputs[1]);
            std::vector<int32_t> ids(inputs[1]->count);
            std::memcpy(ids.data(), raw.data(), raw.size());
            for (int32_t index : ids) require(index >= 0 && index < bound, "Double indexing index out of range");
            if (sorting && output->count) {
                const int inner = p.input_stride(KernelInput::A, axis);
                std::vector<bool> seen(bound);
                for (int row = 0; row < p.output_count() / bound; ++row) {
                    std::fill(seen.begin(), seen.end(), false);
                    const int base = row / inner * bound * inner + row % inner;
                    for (int k = 0; k < bound; ++k) {
                        const int index = ids[base + k * inner];
                        require(!seen[index], "Double sort gradient indices must form a permutation");
                        seen[index] = true;
                    }
                }
            }
        }
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        require(!overlap || (scatter && i == 0 && a == b), "Unsafe overlapping Double indexing operands");
    }
}
} // namespace elman
