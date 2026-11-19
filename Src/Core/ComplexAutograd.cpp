#include "Core/ComplexAutograd.h"
#include "Core/Autograd.h"
#include "Core/TensorDispatch.h"
#include <algorithm>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); }
Shape broadcast_shape(const Shape& a, const Shape& b) {
    Shape shape(std::max(a.size(), b.size()), 1);
    for (size_t d = 0; d < shape.size(); ++d) {
        const int x = d < a.size() ? a[a.size() - 1 - d] : 1;
        const int y = d < b.size() ? b[b.size() - 1 - d] : 1;
        require(x == y || x == 1 || y == 1, "Complex operands cannot broadcast");
        shape[shape.size() - 1 - d] = x == 1 ? y : x;
    }
    return shape;
}
Tensor sum_to(Tensor gradient, const Shape& shape) {
    const DType original = gradient.dtype();
    if (original == DType::Complex32) gradient = cast(gradient, DType::Complex64);
    else if (is_narrow_float(original)) gradient = cast(gradient, DType::Float32);
    while (gradient.shape().size() > shape.size()) gradient = sum(gradient, 0);
    for (size_t d = 0; d < shape.size(); ++d)
        if (shape[d] == 1 && gradient.shape()[d] != 1) gradient = sum(gradient, int(d), true);
    return cast(reshape(gradient, shape), original);
}
DType complex_for_real(DType type) {
    return type == DType::Float64 ? DType::Complex128 :
        type == DType::Float16 ? DType::Complex32 : DType::Complex64;
}
Tensor opmath(const Tensor& x) { return x.dtype() == DType::Complex32 ? cast(x, DType::Complex64) : x; }
Tensor constant(const Tensor& x, double real, double imaginary = 0) {
    return Tensor::from_complex_data({}, {{real, imaginary}}, x.backend(), false, x.dtype());
}
}
DType complex_promoted_dtype(const Tensor& a, const Tensor& b) {
    auto supported = [](DType type) { return is_complex(type) || is_arithmetic_float(type) || is_standard_integer(type); };
    require(supported(a.dtype()) && supported(b.dtype()), "Complex mixed low precision arithmetic requires an explicit cast");
    if (a.dtype() == b.dtype()) return a.dtype();
    if (a.shape().empty() != b.shape().empty()) {
        const auto high = a.shape().empty() ? b.dtype() : a.dtype();
        const auto low = a.shape().empty() ? a.dtype() : b.dtype();
        if (is_complex(high)) return high;
        if (is_complex(low)) return is_arithmetic_float(high) ? complex_for_real(high) : low;
    }
    if (a.dtype() == DType::Complex128 || b.dtype() == DType::Complex128 || a.dtype() == DType::Float64 || b.dtype() == DType::Float64)
        return DType::Complex128;
    if (a.dtype() == DType::Complex64 || b.dtype() == DType::Complex64 || a.dtype() == DType::Float32 || b.dtype() == DType::Float32 ||
        a.dtype() == DType::BFloat16 || b.dtype() == DType::BFloat16) return DType::Complex64;
    return DType::Complex32;
}
Tensor complex(const Tensor& real_part, const Tensor& imaginary_part) {
    require(real_part.dtype() == imaginary_part.dtype() &&
            (real_part.dtype() == DType::Float16 || real_part.dtype() == DType::Float32 || real_part.dtype() == DType::Float64),
            "Complex construction requires matching Half, Float, or Double components");
    const auto shape = broadcast_shape(real_part.shape(), imaginary_part.shape());
    const DType type = complex_for_real(real_part.dtype());
    return graph::tracked(dispatch_tensor("complex", shape, {real_part, imaginary_part}, {}, type),
                          {real_part, imaginary_part}, [real_part, imaginary_part](const Tensor& gradient) {
        if (real_part.requires_grad()) autograd::accumulate(real_part, sum_to(real(gradient), real_part.shape()));
        if (imaginary_part.requires_grad()) autograd::accumulate(imaginary_part, sum_to(imag(gradient), imaginary_part.shape()));
    });
}
Tensor real(const Tensor& x) {
    if (!is_complex(x.dtype())) return x;
    return graph::tracked(dispatch_tensor("real", x.shape(), {x}, {}, complex_component_dtype(x.dtype())), {x},
                          [x](const Tensor& gradient) { autograd::accumulate(x, cast(gradient, x.dtype())); });
}
Tensor imag(const Tensor& x) {
    require(is_complex(x.dtype()), "Imaginary components require complex input");
    return graph::tracked(dispatch_tensor("imag", x.shape(), {x}, {}, complex_component_dtype(x.dtype())), {x},
                          [x](const Tensor& gradient) {
        autograd::accumulate(x, complex(Tensor::zeros(gradient.shape(), gradient.backend(), false, gradient.dtype()), gradient));
    });
}
Tensor conj(const Tensor& x) {
    if (!is_complex(x.dtype())) return x;
    return graph::tracked(dispatch_tensor("conj", x.shape(), {x}, {}, x.dtype()), {x},
                          [x](const Tensor& gradient) { autograd::accumulate(x, conj(gradient)); });
}
Tensor complex_binary(const std::string& operation, const Tensor& left, const Tensor& right) {
    require(operation == "add" || operation == "sub" || operation == "mul" || operation == "div" || operation == "pow",
            "Ordered arithmetic is not defined for complex tensors");
    const DType type = complex_promoted_dtype(left, right);
    const Shape shape = broadcast_shape(left.shape(), right.shape());
    const auto computation = type == DType::Complex32 ? DType::Complex64 : type;
    const Tensor a = cast(left, computation), b = cast(right, computation);
    Tensor result = dispatch_tensor(operation, shape, {a, b}, {}, type);
    return graph::tracked(result, {left, right}, [left, right, a, b, operation](const Tensor& upstream) {
        const auto x = opmath(a), y = opmath(b), g = opmath(upstream);
        Tensor dx, dy;
        if (operation == "add") { dx = g; dy = g; }
        else if (operation == "sub") { dx = g; dy = mul(g, -1.); }
        else if (operation == "mul") { dx = mul(g, conj(y)); dy = mul(g, conj(x)); }
        else if (operation == "div") {
            dx = div(g, conj(y));
            dy = mul(mul(g, -1.), conj(div(div(x, y), y)));
        } else {
            if (left.requires_grad()) dx = where(eq(y, constant(y, 0)), constant(x, 0),
                mul(g, conj(mul(y, pow(x, sub(y, 1.))))));
            if (right.requires_grad()) {
                // The complex exponent derivative masks zero bases for nonnegative real exponents.
                const auto nonnegative = ge(real(y), Tensor::zeros({}, y.backend(), false, complex_component_dtype(y.dtype())));
                dy = where(logical_and(eq(x, constant(x, 0)), nonnegative), constant(x, 0),
                           mul(g, conj(mul(pow(x, y), log(x)))));
            }
        }
        if (left.requires_grad()) autograd::accumulate(left, sum_to(dx, left.shape()));
        if (right.requires_grad()) autograd::accumulate(right, sum_to(dy, right.shape()));
    });
}
Tensor complex_unary(const std::string& operation, const Tensor& input) {
    require(is_complex(input.dtype()), "Complex unary requires complex input");
    require(operation == "sqrt" || operation == "rsqrt" || operation == "exp" || operation == "log" ||
            operation == "sin" || operation == "cos" || operation == "tanh" || operation == "abs" ||
            operation == "angle" || operation == "sgn", "Unary operation is not defined for complex tensors");
    const auto type = operation == "abs" || operation == "angle" ? complex_component_dtype(input.dtype()) : input.dtype();
    Tensor result = dispatch_tensor(operation, input.shape(), {input}, {}, type), saved = result.detach();
    return graph::tracked(result, {input}, [input, saved, operation](const Tensor& upstream) {
        const auto x = opmath(input), g = opmath(upstream);
        const auto y = input.dtype() == DType::Complex32 ? complex_unary(operation, x) : opmath(saved);
        Tensor derivative;
        if (operation == "abs") {
            autograd::accumulate(input, mul(g, sgn(x)));
            return;
        }
        if (operation == "angle") {
            auto squared = add(mul(real(x), real(x)), mul(imag(x), imag(x)));
            derivative = div(mul(x, constant(x, 0, 1)), squared);
            autograd::accumulate(input, where(eq(x, constant(x, 0)), constant(x, 0), mul(g, derivative)));
            return;
        }
        if (operation == "sgn") {
            auto projection = real(mul(conj(y), g));
            auto gradient = div(sub(g, mul(y, projection)), abs(x));
            autograd::accumulate(input, where(eq(x, constant(x, 0)), constant(x, 0), gradient));
            return;
        }
        if (operation == "sqrt") derivative = div(constant(x, .5), y);
        else if (operation == "rsqrt") derivative = mul(pow(y, 3.), -.5);
        else if (operation == "exp") derivative = y;
        else if (operation == "log") derivative = div(constant(x, 1), x);
        else if (operation == "sin") derivative = cos(x);
        else if (operation == "cos") derivative = mul(sin(x), -1.);
        else derivative = sub(constant(x, 1), mul(y, y));
        autograd::accumulate(input, mul(g, conj(derivative)));
    });
}
Tensor angle(const Tensor& x) {
    if (is_complex(x.dtype())) return complex_unary("angle", x);
    require(is_arithmetic_float(x.dtype()), "Real angle requires floating input");
    const auto zero = Tensor::zeros({}, x.backend(), false, x.dtype());
    constexpr double pi_value = 3.14159265358979323846;
    const auto pi = x.dtype() == DType::Float64 ? Tensor::from_double_data({}, {pi_value}, x.backend()) :
        Tensor::from_data({}, {float(pi_value)}, x.backend(), false, x.dtype());
    auto result = where(eq(x, x), where(lt(x, zero), pi, zero), x);
    return graph::tracked(result.detach(), {x}, [x](const Tensor&) {
        autograd::accumulate(x, Tensor::zeros(x.shape(), x.backend(), false, x.dtype()));
    });
}
Tensor sgn(const Tensor& x) { return is_complex(x.dtype()) ? complex_unary("sgn", x) : sign(x); }
Tensor tanh(const Tensor& x) {
    if (is_complex(x.dtype())) return complex_unary("tanh", x);
    require(is_arithmetic_float(x.dtype()), "Tanh requires real floating or complex input");
    return cast(real(complex_unary("tanh", cast(x, complex_for_real(x.dtype())))), x.dtype());
}
Tensor complex_matrix(const Tensor& a, const Tensor& b) {
    require(a.dtype() == b.dtype() && is_complex(a.dtype()), "Complex matrix products require matching dtypes");
    const auto x = opmath(a), y = opmath(b);
    const auto ar = real(x), ai = imag(x), br = real(y), bi = imag(y);
    return cast(complex(sub(matmul(ar, br), matmul(ai, bi)), add(matmul(ar, bi), matmul(ai, br))), a.dtype());
}
Tensor complex_reduce(const std::string& operation, const Tensor& x, int dimension, bool keepdim, bool all) {
    require(operation == "sum" || operation == "mean" || operation == "prod", "Ordered reductions are not defined for complex tensors");
    if (operation != "prod") {
        auto component = [&](const Tensor& value) {
            if (all) return operation == "sum" ? sum_all(value) : mean_all(value);
            return operation == "sum" ? sum(value, dimension, keepdim) : mean(value, dimension, keepdim);
        };
        const auto source = opmath(x);
        auto real_result = component(real(source));
        // An empty CUDA complex mean has a real NaN with a
        // zero imaginary component; CPU initializes both components as NaN.
        auto imaginary_result = operation == "mean" && x.backend() == Backend::CUDA && x.numel() == 0
            ? Tensor::zeros(real_result.shape(), x.backend(), false, real_result.dtype())
            : component(imag(source));
        return cast(complex(real_result, imaginary_result), x.dtype());
    }
    if (x.shape().empty()) {
        require(all || dimension == 0 || dimension == -1, "Scalar complex product axis out of range");
        return reshape(x, {});
    }
    auto values = opmath(x);
    if (all) { values = reshape(values, {int(x.numel())}); dimension = 0; }
    if (dimension < 0) dimension += int(values.shape().size());
    require(dimension >= 0 && dimension < int(values.shape().size()), "Complex product axis out of range");
    int width = values.shape()[dimension];
    if (!width) {
        auto shape = values.shape();
        if (keepdim) shape[dimension] = 1;
        else shape.erase(shape.begin() + dimension);
        return graph::tracked(Tensor::ones(shape, x.backend(), false, x.dtype()), {x}, [x](const Tensor&) {
            autograd::accumulate(x, Tensor::zeros(x.shape(), x.backend(), false, x.dtype()));
        });
    }
    // A balanced backend-native tree keeps work linear and graph depth logarithmic.
    // Differentiating the tree also handles one or several zero factors without division.
    while (width > 1) {
        const int paired = width / 2;
        auto left = slice(values, dimension, 0, paired * 2, 2);
        auto right = slice(values, dimension, 1, paired * 2, 2);
        auto product = mul(left, right);
        values = width % 2 ? concat({product, slice(values, dimension, width - 1, width)}, dimension) : product;
        width = paired + width % 2;
    }
    return cast(keepdim ? values : squeeze(values, dimension), x.dtype());
}
} // namespace elman
