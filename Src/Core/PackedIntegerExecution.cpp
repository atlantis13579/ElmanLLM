#include "Operators/packed_integer_cast.h"
#include "Operators/storage_layout.h"
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace elman {
namespace {
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}
BufferPtr convert(const BufferPtr& input, DType dtype, BufferPtr output = {}) {
    if (!output && input->dtype == dtype) return input;
    if (!output) output = backend_allocate(input->backend, input->count, dtype);
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = int(input->count);
    p.v[KernelParams::kOutputRankSlot] = 1;
    p.v[KernelParams::kOutputShapeSlot] = int(input->count);
    p.v[KernelParams::kOutputDTypeSlot] = int(dtype);
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    const int slot = KernelParams::input_slot(KernelInput::A);
    p.v[slot] = 1;
    p.v[slot + KernelParams::kInputShapeOffset] = int(input->count);
    p.v[slot + KernelParams::kInputStrideOffset] = 1;
    p.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(input->dtype);
    backend_dispatch("cast", output, {input}, p);
    return output;
}
void validate_descriptors(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    require(inputs.size() <= 4 && p.output_dtype() == int(output->dtype), "Packed operation output descriptor mismatch");
    auto shape = [&](int rank, const BufferPtr& buffer, int slot, bool is_output) {
        require(rank >= 0 && rank <= 8, "Invalid packed operation rank");
        size_t count = 1, prefix = 1;
        const size_t limit = size_t(std::numeric_limits<int>::max());
        for (int d = rank - 1; d >= 0; --d) {
            const int length = p.v[slot + d], front = p.v[slot + rank - d - 1];
            require(length >= 0 && front >= 0, "Negative packed operation dimension");
            if (!is_output) require(count <= limit && p.v[slot + 8 + d] == int(count), "Invalid packed operation stride");
            require((!length || count <= limit / size_t(length)) && (!front || prefix <= limit / size_t(front)),
                    "Packed operation dimensions exceed Int32 indexing");
            count *= size_t(length);
            prefix *= size_t(front);
        }
        require(count <= limit && count == buffer->count, "Packed operation shape differs from storage");
    };
    shape(p.output_rank(), output, KernelParams::kOutputShapeSlot, true);
    require(p.output_count() == int(output->count), "Packed operation output count mismatch");
    for (int i = 0; i < 4; ++i) {
        const auto input = KernelInput(i);
        if (size_t(i) >= inputs.size() || !inputs[i]) {
            require(p.input_rank(input) == -1, "Missing packed operation operand");
            continue;
        }
        require(inputs[i]->backend == output->backend && p.input_dtype(input) == int(inputs[i]->dtype),
                "Packed operation backend or dtype mismatch");
        shape(p.input_rank(input), inputs[i], KernelParams::input_slot(input) + KernelParams::kInputShapeOffset, false);
    }
}
}

bool execute_packed_integer(const std::string& name, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    bool packed = is_packed_integer(output->dtype);
    for (const auto& input : inputs) packed = packed || (input && is_packed_integer(input->dtype));
    if (!packed) return false;
    if (storage_layout_operation(name, output->dtype) != StorageLayout::None) return false;
    validate_descriptors(output, inputs, p);
    if (name == "cast") {
        validate_packed_integer_cast(output, inputs, p);
        const auto input = inputs[0];
        if (is_arithmetic_integer(input->dtype) && is_arithmetic_integer(output->dtype)) return false;
        if (is_packed_integer(output->dtype)) {
            const auto decoded = is_low_precision_float(input->dtype) ? convert(input, DType::Float32) : input;
            convert(convert(decoded, packed_integer_opmath(output->dtype)), output->dtype, output);
        } else {
            auto decoded = convert(input, packed_integer_opmath(input->dtype));
            if (is_low_precision_float(output->dtype)) decoded = convert(decoded, DType::Float32);
            convert(decoded, output->dtype, output);
        }
        return true;
    }
    static const std::unordered_set<std::string> supported = {
        "add", "sub", "mul", "pow", "maximum", "minimum", "abs", "sign", "relu", "where",
        "eq", "ne", "gt", "ge", "lt", "le", "logical_and", "logical_or", "logical_not",
        "sum", "prod", "cumsum", "max", "min", "amax", "amin", "matmul", "bmm", "linear", "packed_linear", "addmm",
        "argsort", "scatter", "scatter_add"
    };
    require(supported.count(name) != 0, "Operator does not support packed integer arithmetic");
    const bool logical = name == "logical_and" || name == "logical_or" || name == "logical_not";
    const bool compare = name == "eq" || name == "ne" || name == "gt" || name == "ge" || name == "lt" || name == "le";
    const bool reduce = name == "sum" || name == "prod" || name == "cumsum";
    const bool scatter = name == "scatter" || name == "scatter_add";
    DType common = output->dtype;
    bool found = is_packed_integer(common);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) continue;
        const auto dtype = inputs[i]->dtype;
        if ((name == "where" && i == 0) || (scatter && i == 1)) {
            require(dtype == (scatter ? DType::Int32 : DType::Bool), "Invalid packed indexing or condition dtype");
            continue;
        }
        if (logical) {
            require(is_arithmetic_integer(dtype) || is_arithmetic_float(dtype), "Invalid packed logical operand");
            continue;
        }
        require(is_packed_integer(dtype) && (!found || common == dtype), "Mixed packed integer operands require an explicit cast");
        common = dtype;
        found = true;
    }
    require(logical || found, "Missing packed integer arithmetic operand");
    require(output->dtype == ((logical || compare) ? DType::Bool : reduce ? DType::Int64 :
                             name == "argsort" ? DType::Int32 : common), "Invalid packed integer output dtype");
    if (scatter) {
        require(inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] &&
                (inputs.size() == 3 || !inputs[3]), "Packed scatter requires three operands");
        const int rank = p.output_rank(), axis = p.attribute(0);
        require(rank > 0 && axis >= 0 && axis < rank, "Invalid packed scatter axis");
        for (int i = 0; i < 3; ++i) require(p.input_rank(KernelInput(i)) == rank, "Invalid packed scatter rank");
        for (int d = 0; d < rank; ++d) {
            require(p.output_shape(d) == p.input_shape(KernelInput::A, d), "Packed scatter output shape differs");
            require(p.input_shape(KernelInput::B, d) == p.input_shape(KernelInput::C, d), "Packed scatter source/index shape differs");
            require(d == axis || p.input_shape(KernelInput::C, d) <= p.output_shape(d), "Packed scatter source exceeds output");
        }
    }
    // Int32 opmath preserves exact low-bit arithmetic without floating conversion.
    // Wrap at the final encoding boundary; every temporary remains on its backend.
    auto native_inputs = inputs;
    KernelParams native = p;
    for (size_t i = 0; i < inputs.size(); ++i) if (inputs[i] && is_packed_integer(inputs[i]->dtype)) {
        native_inputs[i] = convert(inputs[i], DType::Int32);
        native.v[KernelParams::input_dtype_slot(KernelInput(i))] = int(DType::Int32);
    }
    const auto destination = is_packed_integer(output->dtype) ? backend_allocate(output->backend, output->count, DType::Int32) : output;
    native.v[KernelParams::kOutputDTypeSlot] = int(destination->dtype);
    backend_dispatch(name, destination, native_inputs, native);
    if (destination != output) convert(destination, output->dtype, output);
    return true;
}
} // namespace elman
