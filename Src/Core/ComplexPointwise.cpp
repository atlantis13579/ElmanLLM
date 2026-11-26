#include "Kernals/complex_pointwise.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace elman {
ComplexPointwise complex_pointwise_operation(const std::string& name, const KernelParams& params) {
    bool complex = is_complex(DType(params.output_dtype()));
    for (int i = 0; i < 3; ++i)
        complex |= params.input_rank(KernelInput(i)) >= 0 && is_complex(DType(params.input_dtype(KernelInput(i))));
    if (!complex) return ComplexPointwise::None;
#define ELMAN_COMPLEX_OP(label, kind) if (name == label) return ComplexPointwise::kind
    ELMAN_COMPLEX_OP("cast", Cast);
    ELMAN_COMPLEX_OP("add", Add);
    ELMAN_COMPLEX_OP("sub", Sub);
    ELMAN_COMPLEX_OP("mul", Mul);
    ELMAN_COMPLEX_OP("div", Div);
    ELMAN_COMPLEX_OP("pow", Pow);
    ELMAN_COMPLEX_OP("conj", Conj);
    ELMAN_COMPLEX_OP("real", Real);
    ELMAN_COMPLEX_OP("imag", Imag);
    ELMAN_COMPLEX_OP("abs", Abs);
    ELMAN_COMPLEX_OP("angle", Angle);
    ELMAN_COMPLEX_OP("sgn", Sgn);
    ELMAN_COMPLEX_OP("sqrt", Sqrt);
    ELMAN_COMPLEX_OP("rsqrt", Rsqrt);
    ELMAN_COMPLEX_OP("exp", Exp);
    ELMAN_COMPLEX_OP("log", Log);
    ELMAN_COMPLEX_OP("sin", Sin);
    ELMAN_COMPLEX_OP("cos", Cos);
    ELMAN_COMPLEX_OP("tanh", Tanh);
    ELMAN_COMPLEX_OP("eq", Eq);
    ELMAN_COMPLEX_OP("ne", Ne);
    ELMAN_COMPLEX_OP("logical_and", LogicalAnd);
    ELMAN_COMPLEX_OP("logical_or", LogicalOr);
    ELMAN_COMPLEX_OP("logical_not", LogicalNot);
    ELMAN_COMPLEX_OP("where", Where);
    ELMAN_COMPLEX_OP("complex", Construct);
#undef ELMAN_COMPLEX_OP
    return ComplexPointwise::None;
}
int complex_pointwise_arity(ComplexPointwise operation) {
    if (operation == ComplexPointwise::Where) return 3;
    if ((operation >= ComplexPointwise::Add && operation <= ComplexPointwise::Pow) ||
        (operation >= ComplexPointwise::Eq && operation <= ComplexPointwise::LogicalOr) ||
        operation == ComplexPointwise::Construct) return 2;
    return 1;
}
void validate_complex_pointwise(ComplexPointwise operation, const BufferPtr& output,
                                 const std::vector<BufferPtr>& inputs, const KernelParams& params) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    const int arity = complex_pointwise_arity(operation);
    require(bool(output) && inputs.size() >= size_t(arity) && inputs.size() <= 4, "Invalid complex pointwise arity");
    require(params.output_dtype() == int(output->dtype), "Complex output dtype descriptor mismatch");
    std::vector<int> shape;
    for (int i = 0; i < 4; ++i) {
        if (i >= arity) {
            require((size_t(i) >= inputs.size() || !inputs[i]) && params.input_rank(KernelInput(i)) == -1,
                    "Unexpected complex pointwise operand");
            continue;
        }
        const KernelInput input = KernelInput(i);
        require(bool(inputs[i]) && inputs[i]->backend == output->backend, "Missing or mixed backend complex operand");
        const DType type = inputs[i]->dtype;
        require(params.input_dtype(input) == int(type), "Complex input dtype descriptor mismatch");
        require(is_complex(type) || is_arithmetic_float(type) || is_standard_integer(type), "Unsupported complex operand dtype");
        if (operation == ComplexPointwise::Where && i == 0) require(type == DType::Bool, "Complex where requires Boolean condition");
        if (operation == ComplexPointwise::Construct)
            require(is_arithmetic_float(type) && type == complex_component_dtype(output->dtype), "Complex construction component dtype mismatch");
        const int rank = params.input_rank(input);
        require(rank >= 0 && rank <= 8, "Invalid complex operand rank");
        std::vector<int> dimensions(rank);
        size_t count = 1, prefix = 1;
        for (int d = rank - 1; d >= 0; --d) {
            dimensions[d] = params.input_shape(input, d);
            require(dimensions[d] >= 0 && count <= size_t(INT_MAX) && params.input_stride(input, d) == int(count),
                    "Invalid complex contiguous descriptor");
            require(!dimensions[d] || count <= size_t(INT_MAX) / dimensions[d], "Complex suffix exceeds indexing limit");
            count *= size_t(dimensions[d]);
        }
        for (int length : dimensions) {
            require(!length || prefix <= size_t(INT_MAX) / length, "Complex prefix exceeds indexing limit");
            prefix *= size_t(length);
        }
        require(count == inputs[i]->count, "Complex input shape differs from storage");
        std::vector<int> combined(std::max(shape.size(), dimensions.size()), 1);
        for (size_t d = 0; d < combined.size(); ++d) {
            const int a = d < shape.size() ? shape[shape.size() - 1 - d] : 1;
            const int b = d < dimensions.size() ? dimensions[dimensions.size() - 1 - d] : 1;
            require(a == b || a == 1 || b == 1, "Complex operands cannot broadcast");
            combined[combined.size() - 1 - d] = a == 1 ? b : a;
        }
        shape = std::move(combined);
    }
    require(params.output_rank() == int(shape.size()), "Complex output rank mismatch");
    size_t count = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        require(params.output_shape(int(d)) == shape[d], "Complex output shape mismatch");
        require(!shape[d] || count <= size_t(INT_MAX) / shape[d], "Complex output exceeds indexing limit");
        count *= size_t(shape[d]);
    }
    require(count == output->count && params.output_count() == int(count), "Complex output count mismatch");
    const bool boolean = operation >= ComplexPointwise::Eq && operation <= ComplexPointwise::LogicalNot;
    const bool component = operation == ComplexPointwise::Real || operation == ComplexPointwise::Imag ||
                           operation == ComplexPointwise::Abs || operation == ComplexPointwise::Angle;
    if (operation == ComplexPointwise::Cast)
        require(is_complex(output->dtype) || is_arithmetic_float(output->dtype) || output->dtype == DType::Bool,
                "Complex integer conversion requires an explicit real-component conversion");
    else if (boolean) require(output->dtype == DType::Bool, "Complex comparison output must be Boolean");
    else if (component) require(is_complex(inputs[0]->dtype) && output->dtype == complex_component_dtype(inputs[0]->dtype),
                                "Complex component output dtype mismatch");
    else require(is_complex(output->dtype), "Complex arithmetic requires complex output");
    for (int i = 0; i < arity; ++i) {
        if (!output->bytes || !inputs[i]->bytes) continue;
        const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer());
        const uintptr_t b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
        const bool overlaps = a <= b ? b - a < output->bytes : a - b < inputs[i]->bytes;
        if (!overlaps) continue;
        require(a == b && output->dtype == inputs[i]->dtype && output->count == inputs[i]->count &&
                params.input_rank(KernelInput(i)) == params.output_rank(), "Unsafe overlapping complex buffers");
        for (size_t d = 0; d < shape.size(); ++d)
            require(params.input_shape(KernelInput(i), int(d)) == shape[d], "Overlapping complex broadcast operand");
    }
}
} // namespace elman
