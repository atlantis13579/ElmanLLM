#include "Operators/storage_layout.h"
#include "Core/LocalFunction.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace elman {
void validate_storage_layout(StorageLayout op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    // Indexed copy has its own validated scalar normalization, duplicate and alias rules.
    if (op == StorageLayout::IndexCopy) return;
    const int arity = op == StorageLayout::Concat || op == StorageLayout::Gather || op == StorageLayout::IndexSelect ? 2 : 1;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid storage layout operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected storage layout operand");
    const bool low = is_low_precision_float(output->dtype) || is_complex(output->dtype);
    if (low) {
        require(p.output_dtype() == int(output->dtype), "Storage output dtype differs from its descriptor");
        for (int i = arity; i < 4; ++i)
            require(p.input_rank(KernelInput(i)) == -1, "Unexpected storage input descriptor");
    }
    LocalFunction shape_count{[&](const std::vector<int>& shape) {
        size_t forward = 1, backward = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            const int a = shape[i], b = shape[shape.size() - i - 1];
            require(a >= 0 && b >= 0, "Negative storage layout dimension");
            require((a == 0 || forward <= size_t(INT32_MAX) / size_t(a)) &&
                    (b == 0 || backward <= size_t(INT32_MAX) / size_t(b)), "Storage layout dimensions exceed Int32 indexing");
            forward *= size_t(a); backward *= size_t(b);
        }
        return forward;
    }};
    std::vector<std::vector<int>> shape(arity);
    for (int i = 0; i < arity; ++i) {
        require(bool(inputs[i]), "Missing storage layout operand");
        const KernelInput input = KernelInput(i);
        const int rank = p.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid storage layout input rank");
        shape[i].resize(rank);
        for (int d = 0; d < rank; ++d) shape[i][d] = p.input_shape(input, d);
        require(shape_count(shape[i]) == inputs[i]->count, "Storage layout input shape differs from storage");
        int stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            require(p.input_stride(input, d) == stride, "Invalid storage layout contiguous stride");
            stride *= shape[i][d];
        }
        const bool index = i == 1 && (op == StorageLayout::Gather || op == StorageLayout::IndexSelect);
        require(inputs[i]->dtype == (index ? DType::Int32 : output->dtype), "Invalid storage layout input dtype");
        if (low) require(p.input_dtype(input) == int(inputs[i]->dtype), "Storage input dtype differs from its descriptor");
    }
    const int rank = p.output_rank();
    require(rank >= 0 && rank <= 8, "Invalid storage layout output rank");
    std::vector<int> target(rank);
    for (int d = 0; d < rank; ++d) target[d] = p.output_shape(d);
    require(shape_count(target) == output->count && p.output_count() == int(output->count), "Storage layout output shape differs from storage");
    const std::vector<int>& source = shape[0];
    const int source_rank = int(source.size()), axis = p.attribute(0);
    if (op == StorageLayout::Copy) {
        require(inputs[0]->count == output->count, "Storage reshape changes element count");
    } else if (op == StorageLayout::Permute || op == StorageLayout::Transpose) {
        require(rank == source_rank, "Storage permutation changes rank");
        if (op == StorageLayout::Transpose)
            require(rank ? axis >= 0 && axis < rank && p.attribute(1) >= 0 && p.attribute(1) < rank :
                    (axis == -1 || axis == 0) && (p.attribute(1) == -1 || p.attribute(1) == 0), "Invalid storage transpose axes");
        std::vector<bool> seen(rank);
        for (int d = 0; d < rank; ++d) {
            const int selected = op == StorageLayout::Permute ? p.attribute(d) : d == axis ? p.attribute(1) : d == p.attribute(1) ? axis : d;
            require(selected >= 0 && selected < rank && !seen[selected], "Invalid storage permutation axes");
            seen[selected] = true;
            require(target[d] == source[selected], "Storage permutation output shape differs");
        }
    } else if (op == StorageLayout::Expand || op == StorageLayout::Repeat) {
        require(rank >= source_rank, "Storage broadcast removes dimensions");
        for (int d = 0; d < source_rank; ++d) {
            const int from = source[d], to = target[d + rank - source_rank];
            if (op == StorageLayout::Expand) require(from == to || from == 1, "Invalid storage expand dimensions");
            else require(from == 0 ? to == 0 : to % from == 0, "Invalid storage repeat dimensions");
        }
    } else if (op == StorageLayout::Triu || op == StorageLayout::Tril) {
        require(rank >= 2 && target == source, "Invalid triangular storage shape");
    } else {
        require(rank == source_rank && axis >= 0 && axis < rank, "Invalid storage layout axis/rank");
        if (op == StorageLayout::Slice) {
            const int start = p.attribute(1), step = p.attribute(2);
            require(start >= 0 && start <= source[axis] && step > 0, "Invalid storage slice parameters");
            require(target[axis] == 0 || int64_t(start) + int64_t(target[axis] - 1) * step < source[axis], "Storage slice exceeds its input");
        } else if (op == StorageLayout::SliceGradient) {
            const int start = p.attribute(1), step = p.attribute(2);
            require((is_complex(output->dtype) || output->dtype == DType::Float64 || is_low_precision_float(output->dtype)) && start >= 0 && start <= target[axis] && step > 0,
                    "Invalid slice gradient parameters");
            require(source[axis] == 0 || int64_t(start) + int64_t(source[axis] - 1) * step < target[axis],
                    "Slice gradient exceeds its output");
        } else if (op == StorageLayout::Concat) {
            require(int(shape[1].size()) == rank && int64_t(source[axis]) + shape[1][axis] == target[axis], "Invalid storage concat length");
        } else if (op == StorageLayout::Gather) {
            require(target == shape[1], "Storage gather output must match indices");
        } else if (op == StorageLayout::IndexSelect) {
            require(shape[1].size() == 1 && target[axis] == int(inputs[1]->count), "Invalid storage selection indices");
        }
        for (int d = 0; d < rank; ++d) if (d != axis) {
            require(op == StorageLayout::Gather ? target[d] <= source[d] : target[d] == source[d], "Storage layout non-axis dimensions differ");
            if (op == StorageLayout::Concat) require(shape[1][d] == source[d], "Storage concat non-axis dimensions differ");
        }
        if ((op == StorageLayout::Gather || op == StorageLayout::IndexSelect) && output->count) {
            const std::vector<uint8_t> bytes = backend_download_raw(inputs[1]);
            for (size_t i = 0; i < inputs[1]->count; ++i) {
                int32_t selected;
                std::memcpy(&selected, bytes.data() + i * 4, 4);
                require(selected >= 0 && selected < source[axis], "Storage layout index out of range");
            }
        }
    }
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        require(!overlap || (op == StorageLayout::Copy && a == b), "Storage layout output overlaps an input");
    }
}
} // namespace elman
