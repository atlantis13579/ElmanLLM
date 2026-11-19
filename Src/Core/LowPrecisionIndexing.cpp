#include "Operators/low_precision_indexing.h"
#include "Core/Memory.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace elman {
void validate_low_precision_indexing(LowPrecisionIndexing op, const BufferPtr& output,
                               const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool repeat = op == LowPrecisionIndexing::RepeatGradient;
    const bool scatter = op == LowPrecisionIndexing::Scatter || op == LowPrecisionIndexing::ScatterAdd;
    const bool source_gradient = op == LowPrecisionIndexing::ScatterGradient;
    const bool selection = op == LowPrecisionIndexing::SelectionGradient;
    const bool sorting = op == LowPrecisionIndexing::SortGradient;
    const int arity = repeat ? 1 : scatter ? 3 : 2;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid low precision indexing operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected low precision indexing operand");
    for (int i = arity; i < 4; ++i) require(p.input_rank(KernelInput(i)) == -1, "Unexpected indexing descriptor");
    require(p.output_dtype() == int(output->dtype), "Indexing output dtype differs");
    require(is_low_precision_float(output->dtype), "Low precision indexing requires a low precision output");
    auto shape = [&](int rank, auto dimension, auto stride, size_t expected) {
        require(rank >= 0 && rank <= 8, "Invalid low precision indexing rank");
        std::vector<int> result(rank);
        size_t forward = 1, backward = 1;
        for (int d = 0; d < rank; ++d) {
            const int length = dimension(d), reverse = dimension(rank - d - 1);
            require(length >= 0 && reverse >= 0 &&
                    (length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
                    (reverse == 0 || backward <= size_t(INT32_MAX) / size_t(reverse)), "Low precision indexing dimensions overflow");
            require(stride(rank - d - 1, int(backward)), "Invalid low precision indexing contiguous stride");
            forward *= size_t(length); backward *= size_t(reverse); result[d] = length;
        }
        require(forward == expected, "Low precision indexing shape differs from storage");
        return result;
    };
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == (i == 1 ? DType::Int32 : output->dtype), "Invalid low precision indexing operand dtype");
        require(p.input_dtype(KernelInput(i)) == int(inputs[i]->dtype), "Indexing descriptor dtype differs");
        const auto input = KernelInput(i);
        shapes.push_back(shape(p.input_rank(input), [&](int d) { return p.input_shape(input, d); },
            [&](int d, int value) { return p.input_stride(input, d) == value; }, inputs[i]->count));
    }
    auto target = shape(p.output_rank(), [&](int d) { return p.output_shape(d); }, [](int, int) { return true; }, output->count);
    require(p.output_count() == int(output->count), "Invalid low precision indexing output count");
    const auto& source = shapes[0];
    const int rank = int(target.size()), axis = p.attribute(0);
    int bound = 0;
    if (repeat) {
        require(source.size() >= target.size(), "Low precision repeat gradient removes source dimensions");
        for (int d = 0; d < rank; ++d) {
            const int from = target[d], to = source[d + source.size() - target.size()];
            require(from ? to % from == 0 : to == 0, "Invalid low precision repeat gradient shape");
        }
    } else {
        require(rank >= 1 && int(source.size()) == rank && axis >= 0 && axis < rank, "Invalid low precision indexing axis/rank");
        const auto& indexed = scatter ? shapes[2] : source_gradient ? target : source;
        require(int(indexed.size()) == rank, "Low precision indexing operand ranks differ");
        if (selection) require(shapes[1].size() == 1 && indexed[axis] == int(inputs[1]->count), "Invalid low precision selection indices");
        else require(shapes[1] == indexed, "Low precision indexing indices must match the selected values");
        if (scatter) require(source == target, "Low precision scatter destination shape differs");
        if (sorting) require(source == target, "Low precision sort gradient shapes differ");
        const auto& destination = source_gradient ? source : target;
        for (int d = 0; d < rank; ++d) if (d != axis)
            require(selection || sorting ? indexed[d] == destination[d] : indexed[d] <= destination[d], "Low precision indexing non-axis dimensions differ");
        bound = destination[axis];
        if (inputs[1]->count) {
            MemoryReservation host(Backend::CPU, checked_memory_add(checked_memory_mul(inputs[1]->bytes, 2), sorting ? size_t(bound) : 0));
            const auto raw = backend_download_raw(inputs[1]);
            std::vector<int32_t> ids(inputs[1]->count);
            std::memcpy(ids.data(), raw.data(), raw.size());
            for (int32_t index : ids) require(index >= 0 && index < bound, "Low precision indexing index out of range");
            if (sorting && output->count) {
                const int inner = p.input_stride(KernelInput::A, axis);
                std::vector<bool> seen(bound);
                for (int row = 0; row < p.output_count() / bound; ++row) {
                    std::fill(seen.begin(), seen.end(), false);
                    const int base = row / inner * bound * inner + row % inner;
                    for (int k = 0; k < bound; ++k) {
                        const int index = ids[base + k * inner];
                        require(!seen[index], "Low precision sort gradient indices must form a permutation");
                        seen[index] = true;
                    }
                }
            }
        }
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        require(!overlap, "Unsafe overlapping low precision indexing operands");
    }
}
} // namespace elman
