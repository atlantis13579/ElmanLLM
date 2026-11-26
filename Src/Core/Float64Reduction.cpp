#include "Kernals/float64_reduction.h"
#include <stdexcept>

namespace elman {
void validate_float64_reduction(Float64Reduction op, const BufferPtr& output,
                                const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const bool gradient = op == Float64Reduction::Gradient;
    const bool scan = op == Float64Reduction::Cumsum || op == Float64Reduction::ReverseCumsum;
    const int arity = gradient ? 3 : 1;
    require(inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid Float64 reduction operand count");
    for (size_t i = arity; i < inputs.size(); ++i) require(!inputs[i], "Unexpected Float64 reduction operand");
    require(output->dtype == DType::Float64, "Float64 reduction output dtype differs");
    std::vector<std::vector<int>> shapes;
    for (int i = 0; i < arity; ++i) {
        require(inputs[i] && inputs[i]->dtype == DType::Float64, "Missing or mistyped Float64 reduction input");
        const KernelInput input = KernelInput(i);
        const int rank = p.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid Float64 reduction rank");
        std::vector<int> shape(rank);
        size_t count = 1, prefix = 1;
        for (int d = rank - 1; d >= 0; --d) {
            shape[d] = p.input_shape(input, d);
            require(shape[d] >= 0 && count <= size_t(INT32_MAX) && p.input_stride(input, d) == int(count),
                    "Invalid Float64 reduction contiguous descriptor");
            require(shape[d] == 0 || count <= size_t(INT32_MAX) / size_t(shape[d]), "Float64 reduction dimensions overflow");
            count *= size_t(shape[d]);
        }
        for (int length : shape) {
            require(length == 0 || prefix <= size_t(INT32_MAX) / size_t(length), "Float64 reduction prefix dimensions overflow");
            prefix *= size_t(length);
        }
        require(count == inputs[i]->count, "Float64 reduction input count differs");
        shapes.push_back(std::move(shape));
    }
    const int axis = p.attribute(0), rank = int(shapes[0].size());
    require(scan ? (rank == 0 ? axis == 0 : axis >= 0 && axis < rank) : axis >= -1 && axis < rank,
            "Invalid Float64 reduction axis");
    require(scan || p.attribute(1) == 0 || p.attribute(1) == 1, "Invalid Float64 reduction keepdim");
    require(!gradient || (p.attribute(2) >= 0 && p.attribute(2) <= 6), "Invalid Float64 reduction gradient kind");
    std::vector<int> reduced = shapes[0];
    if (!scan) {
        if (axis < 0) reduced = p.attribute(1) ? std::vector<int>(rank, 1) : std::vector<int>{};
        else if (p.attribute(1)) reduced[axis] = 1;
        else reduced.erase(reduced.begin() + axis);
    }
    const bool extremum = op == Float64Reduction::Maximum || op == Float64Reduction::Minimum ||
        (gradient && (p.attribute(2) == 2 || p.attribute(2) == 3 || p.attribute(2) == 5 || p.attribute(2) == 6));
    require(!extremum || (axis < 0 ? inputs[0]->count > 0 : shapes[0][axis] > 0), "Float64 extremum requires a nonempty axis");
    if (gradient) require(shapes[1] == reduced && shapes[2] == reduced, "Float64 reduction saved output or gradient shape differs");
    const std::vector<int>& expected = gradient || scan ? shapes[0] : reduced;
    require(p.output_rank() == int(expected.size()), "Invalid Float64 reduction output rank");
    size_t count = 1;
    for (size_t d = 0; d < expected.size(); ++d) {
        require(p.output_shape(int(d)) == expected[d], "Invalid Float64 reduction output shape");
        require(expected[d] == 0 || count <= size_t(INT32_MAX) / size_t(expected[d]), "Float64 reduction output count overflow");
        count *= size_t(expected[d]);
    }
    require(count == output->count && p.output_count() == int(count), "Float64 reduction output count differs");
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const uintptr_t first = reinterpret_cast<uintptr_t>(output->raw_pointer());
        const uintptr_t second = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlap = first <= second ? second - first < output->bytes : first - second < inputs[i]->bytes;
        require(!overlap || (scan && first == second && output->bytes == inputs[i]->bytes), "Unsafe overlapping Float64 reduction buffers");
    }
}
} // namespace elman
