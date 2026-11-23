#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "Core/Autograd.h"
#include "Core/LocalFunction.h"
#include "Core/Autocast.h"
#include "Core/ComplexAutograd.h"
#include "Core/LowPrecisionFloat.h"
#include "Core/TensorDispatch.h"
#include "Core/Memory.h"
#include "Operators/group_norm_params.h"
#include "Operators/attributes.h"
#include "Operators/integer_matrix.h"
#include "Operators/low_precision_attention.h"

namespace elman {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

int axis(int dimension, size_t rank) {
  if (dimension < 0) {
    dimension += static_cast<int>(rank);
  }
  require(dimension >= 0 && dimension < static_cast<int>(rank),
          "Axis is out of range");
  return dimension;
}

int float_bits(float value) {
  int bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}  // namespace

using autograd::accumulate;
using graph::tracked;

static Shape broadcast(const Shape& a, const Shape& b) {
  Shape out(std::max(a.size(), b.size()), 1);
  for (size_t k = 0; k < out.size(); ++k) {
    int x = k < a.size() ? a[a.size() - 1 - k] : 1,
        y = k < b.size() ? b[b.size() - 1 - k] : 1;
    require(x == y || x == 1 || y == 1, "Shapes cannot broadcast");
    out[out.size() - 1 - k] = x == 1 ? y : x;
  }
  return out;
}
static Tensor unbroadcast(Tensor g, const Shape& shape) {
  if (is_narrow_float(g.dtype())) {
    int leading = int(g.shape().size()) - int(shape.size());
    std::vector<int> retained, reduced;
    size_t count = 1;
    for (int d = 0; d < int(g.shape().size()); ++d) {
      if (d < leading || (shape[d - leading] == 1 && g.shape()[d] != 1)) {
        reduced.push_back(d);
        count *= size_t(g.shape()[d]);
      } else retained.push_back(d);
    }
    if (reduced.size() > 1) {
      // sum_to reduces all broadcast axes together in opmath precision. Separate
      // narrow reductions would round the partial sums between dimensions.
      retained.insert(retained.end(), reduced.begin(), reduced.end());
      Tensor rows = reshape(permute(g, retained), {int(shape_element_count(shape)), int(count)});
      return reshape(sum(rows, 1), shape);
    }
  }
  while (g.shape().size() > shape.size()) {
    g = sum(g, 0);
  }
  for (int d = 0; d < int(shape.size()); ++d) {
    if (shape[d] == 1 && g.shape()[d] != 1) {
      g = sum(g, d, true);
    }
  }
  return reshape(g, shape);
}
static Tensor scalar(float v, Backend b) {
  return Tensor::from_data({}, {v}, b);
}
static DType promoted_dtype(const Tensor& a, const Tensor& b) {
  if (is_complex(a.dtype()) || is_complex(b.dtype())) return complex_promoted_dtype(a, b);
  if (a.dtype() == b.dtype()) return a.dtype();
  require(!is_low_precision_float(a.dtype()) && !is_low_precision_float(b.dtype()),
          "Mixed low precision arithmetic requires an explicit cast");
  require(!is_packed_integer(a.dtype()) && !is_packed_integer(b.dtype()),
          "Mixed packed integer arithmetic requires an explicit cast");
  int (*const category)(DType) = [](DType dtype) {
    return is_arithmetic_float(dtype) || dtype == DType::Float64 ? 2 :
        dtype == DType::Bool ? 0 : is_arithmetic_integer(dtype) ? 1 : -1;
  };
  require(category(a.dtype()) >= 0 && category(b.dtype()) >= 0, "Unsupported promotion dtype");
  if (a.shape().empty() != b.shape().empty()) {
    const Tensor& dimensional = a.shape().empty() ? b : a;
    const Tensor& zero_dimensional = a.shape().empty() ? a : b;
    if (category(dimensional.dtype()) >= category(zero_dimensional.dtype())) return dimensional.dtype();
  }
  if (category(a.dtype()) == 2 || category(b.dtype()) == 2) {
    if (a.dtype() == DType::Float64 || b.dtype() == DType::Float64) return DType::Float64;
    if (a.dtype() == DType::Float32 || b.dtype() == DType::Float32) return DType::Float32;
    if (is_arithmetic_float(a.dtype()) && is_arithmetic_float(b.dtype())) return DType::Float32;
    return is_arithmetic_float(a.dtype()) ? a.dtype() : b.dtype();
  }
  bool (*const wide_unsigned)(DType) = [](DType dtype) {
    return dtype == DType::UInt16 || dtype == DType::UInt32 || dtype == DType::UInt64;
  };
  require(!wide_unsigned(a.dtype()) && !wide_unsigned(b.dtype()), "Mixed unsigned 16/32/64-bit promotion is undefined");
  if (a.dtype() == DType::Bool) return b.dtype();
  if (b.dtype() == DType::Bool) return a.dtype();
  if (a.dtype() == DType::Int64 || b.dtype() == DType::Int64) return DType::Int64;
  if (a.dtype() == DType::Int32 || b.dtype() == DType::Int32) return DType::Int32;
  if (a.dtype() == DType::Int16 || b.dtype() == DType::Int16 || a.dtype() == DType::UInt8 || b.dtype() == DType::UInt8)
    return DType::Int16;
  return DType::Int8;
}
Tensor where(const Tensor& condition, const Tensor& a, const Tensor& b) {
  require(condition.dtype() == DType::Bool, "where condition must be bool");
  const DType dtype = promoted_dtype(a, b);
  Shape shape = broadcast(condition.shape(), broadcast(a.shape(), b.shape()));
  const DType computation = dtype == DType::Complex32 ? DType::Complex64 : dtype;
  return tracked(dispatch_tensor("where", shape, {condition, cast(a, computation), cast(b, computation)}, {}, dtype),
                 {a, b}, [condition, a, b](const Tensor& g) {
                   const Tensor gradient = is_low_precision_float(g.dtype()) ? cast(g, DType::Float32) : g;
                   Tensor zero = scalar(0, gradient.backend());
                   if (a.requires_grad()) accumulate(a, unbroadcast(where(condition, gradient, zero), a.shape()));
                   if (b.requires_grad()) accumulate(b, unbroadcast(where(condition, zero, gradient), b.shape()));
                 });
}
static Tensor scalar_binary(const std::string& op, const Tensor& a, double argument) {
  if (is_complex(a.dtype())) {
    const DType computation = a.dtype() == DType::Complex32 ? DType::Complex64 : a.dtype();
    return cast(complex_binary(op, cast(a, computation),
        Tensor::from_complex_data({}, {{argument, 0}}, a.backend(), false, computation)), a.dtype());
  }
  const bool wide = a.dtype() == DType::Float64;
  const float value = float(argument);
  require(op != "sub" || a.dtype() != DType::Bool, "Subtraction does not support bool operands");
  const DType dtype = is_differentiable_float(a.dtype()) ? a.dtype() : DType::Float32;
  const bool reciprocal = op == "div" && a.backend() == Backend::CUDA && dtype != DType::Float64;
  Tensor right = wide ? Tensor::from_double_data({}, {argument}, a.backend()) :
      scalar(reciprocal ? 1.0f / value : value, a.backend());
  // Wrapped numeric scalars differ from ordinary zero-dimensional tensors.
  // CPU add/sub and both power kernels convert the scalar to the result dtype;
  // CUDA add/sub and multiplication/division retain FP32 scalar precision.
  if (!is_low_precision_float(dtype) && (op == "pow" || (a.backend() == Backend::CPU && (op == "add" || op == "sub"))))
    right = cast(right, dtype);
  return tracked(dispatch_tensor(reciprocal ? "mul" : op, a.shape(), {cast(a, dtype), right}, {}, dtype), {a},
                 [original = a, value, argument, wide, op](const Tensor& upstream) {
    const bool low = is_low_precision_float(original.dtype());
    const Tensor a = low ? cast(original, DType::Float32) : original;
    const Tensor g = low ? cast(upstream, DType::Float32) : upstream;
    const double factor = wide ? argument : double(value);
    if (op == "add" || op == "sub") accumulate(original, g);
    else if (op == "mul") accumulate(original, mul(g, factor));
    else if (op == "div") accumulate(original, div(g, factor));
    else if (factor == 0) accumulate(original, Tensor::zeros(a.shape(), a.backend(), false, a.dtype()));
    else if (wide) accumulate(original, mul(g, mul(pow(a, factor - 1.), factor)));
    else accumulate(original, mul(g, mul(pow(a, value - 1), value)));
  });
}
Tensor add(const Tensor& a, double b) { return scalar_binary("add", a, b); }
Tensor sub(const Tensor& a, double b) { return scalar_binary("sub", a, b); }
Tensor mul(const Tensor& a, double b) { return scalar_binary("mul", a, b); }
Tensor div(const Tensor& a, double b) { return scalar_binary("div", a, b); }
Tensor pow(const Tensor& a, double b) { return scalar_binary("pow", a, b); }
static Tensor binary(const std::string& op, const Tensor& a, const Tensor& b) {
  if (is_complex(a.dtype()) || is_complex(b.dtype())) return complex_binary(op, a, b);
  Shape s = broadcast(a.shape(), b.shape());
  DType dtype = promoted_dtype(a, b);
  if (op == "div" && !is_differentiable_float(dtype)) dtype = DType::Float32;
  require(op != "sub" || (a.dtype() != DType::Bool && b.dtype() != DType::Bool), "Subtraction does not support bool operands");
  require(op != "pow" || dtype != DType::Bool, "Power does not support bool operands");
  // Retain Float32 scalar precision for narrow CPU multiplication and division.
  const bool opmath_scalar = a.backend() == Backend::CPU && is_narrow_float(dtype) &&
                            (op == "mul" || op == "div") && !a.shape().empty() &&
                            b.shape().empty() && b.dtype() == DType::Float32;
  Tensor out = dispatch_tensor(op, s, {cast(a, dtype), opmath_scalar ? b : cast(b, dtype)}, {}, dtype);
  return tracked(out, {a, b}, [left = a, right = b, op](const Tensor& upstream) {
    const bool low = is_low_precision_float(left.dtype()) || is_low_precision_float(right.dtype());
    const Tensor a = low ? cast(left, DType::Float32) : left;
    const Tensor b = low ? cast(right, DType::Float32) : right;
    const Tensor g = low ? cast(upstream, DType::Float32) : upstream;
    Tensor ga, gb;
    if (op == "add") {
      ga = g;
      gb = g;
    } else if (op == "sub") {
      ga = g;
      gb = mul(g, -1);
    } else if (op == "mul") {
      ga = mul(g, b);
      gb = mul(g, a);
    } else if (op == "div") {
      ga = div(g, b);
      gb = mul(mul(g, -1), div(div(a, b), b));
    } else if (op == "pow") {
      if (left.requires_grad()) {
        ga = where(eq(b, scalar(0, b.backend())), scalar(0, a.backend()),
                   mul(g, mul(b, pow(a, sub(b, 1)))));
      }
      if (right.requires_grad()) {
        gb = where(logical_and(eq(a, scalar(0, a.backend())), ge(b, scalar(0, b.backend()))),
                   scalar(0, a.backend()), mul(g, mul(pow(a, b), log(a))));
      }
    } else {
      bool mx = op == "maximum";
      Tensor shared = where(eq(a, b), mul(g, 0.5f), g);
      ga = where(mx ? lt(a, b) : gt(a, b), scalar(0, g.backend()), shared);
      gb = where(mx ? gt(a, b) : lt(a, b), scalar(0, g.backend()), shared);
    }
    if (left.requires_grad()) {
      accumulate(left, unbroadcast(ga, a.shape()));
    }
    if (right.requires_grad()) {
      accumulate(right, unbroadcast(gb, b.shape()));
    }
  });
}
Tensor add(const Tensor& a, const Tensor& b) {
  return binary("add", a, b);
}
Tensor sub(const Tensor& a, const Tensor& b) {
  return binary("sub", a, b);
}
Tensor mul(const Tensor& a, const Tensor& b) {
  return binary("mul", a, b);
}
Tensor div(const Tensor& a, const Tensor& b) {
  return binary("div", a, b);
}
Tensor pow(const Tensor& a, const Tensor& b) {
  return binary("pow", a, b);
}
Tensor maximum(const Tensor& a, const Tensor& b) {
  return binary("maximum", a, b);
}
Tensor minimum(const Tensor& a, const Tensor& b) {
  return binary("minimum", a, b);
}

static Tensor comparison(const std::string& op, const Tensor& a, const Tensor& b) {
  require((!is_complex(a.dtype()) && !is_complex(b.dtype())) || op == "eq" || op == "ne",
          "Ordered comparison is not defined for complex tensors");
  const DType dtype = promoted_dtype(a, b);
  const DType computation = dtype == DType::Complex32 ? DType::Complex64 : dtype;
  return dispatch_tensor(op, broadcast(a.shape(), b.shape()), {cast(a, computation), cast(b, computation)}, {}, DType::Bool);
}
Tensor eq(const Tensor& a, const Tensor& b) {
  return comparison("eq", a, b);
}
Tensor ne(const Tensor& a, const Tensor& b) {
  return comparison("ne", a, b);
}
Tensor gt(const Tensor& a, const Tensor& b) {
  return comparison("gt", a, b);
}
Tensor ge(const Tensor& a, const Tensor& b) {
  return comparison("ge", a, b);
}
Tensor lt(const Tensor& a, const Tensor& b) {
  return comparison("lt", a, b);
}
Tensor le(const Tensor& a, const Tensor& b) {
  return comparison("le", a, b);
}
Tensor logical_and(const Tensor& a, const Tensor& b) {
  const DType dtype = promoted_dtype(a, b);
  const DType computation = dtype == DType::Complex32 ? DType::Complex64 : dtype;
  return dispatch_tensor("logical_and", broadcast(a.shape(), b.shape()), {cast(a, computation), cast(b, computation)},
                         {}, DType::Bool);
}
Tensor logical_or(const Tensor& a, const Tensor& b) {
  const DType dtype = promoted_dtype(a, b);
  const DType computation = dtype == DType::Complex32 ? DType::Complex64 : dtype;
  return dispatch_tensor("logical_or", broadcast(a.shape(), b.shape()), {cast(a, computation), cast(b, computation)},
                         {}, DType::Bool);
}

Tensor logical_not(const Tensor& x) {
  return dispatch_tensor("logical_not", x.shape(), {x}, {}, DType::Bool);
}
Tensor cast(const Tensor& x, DType dtype) {
  if (x.dtype() == dtype) return x;
  if (is_complex(x.dtype()) && (is_arithmetic_integer(dtype) && dtype != DType::Bool || is_low_precision_float(dtype)))
    return cast(real(x), dtype);
  if (is_complex(dtype) && (is_packed_integer(x.dtype()) || is_low_precision_float(x.dtype())))
    return cast(cast(x, DType::Float32), dtype);
  Tensor y = dispatch_tensor("cast", x.shape(), {x}, {}, dtype);
  if ((is_differentiable_float(x.dtype()) || is_complex(x.dtype())) && (is_differentiable_float(dtype) || is_complex(dtype))) {
    return tracked(y, {x}, [x](const Tensor& g) { accumulate(x, g); });
  }
  return y;
}
static Tensor unary(const std::string& op, const Tensor& x,
                    double slope = 0.01) {
  if (is_complex(x.dtype())) return complex_unary(op, x);
  require(x.dtype() != DType::Bool || (op != "relu" && (op != "abs" || x.backend() == Backend::CUDA)), "Activation does not support bool input");
  require(is_differentiable_float(x.dtype()) || (op != "silu" && op != "gelu" && op != "leaky_relu"), "Activation requires floating input");
  const bool preserve = op == "abs" || op == "sign" || op == "relu";
  const Tensor input = !preserve && is_arithmetic_integer(x.dtype()) ? cast(x, DType::Float32) : x;
  std::vector<int> attributes;
  if (op == "leaky_relu" && x.dtype() == DType::Float64) {
    attributes.resize(3);
    std::memcpy(attributes.data(), &slope, sizeof(slope));
    attributes[2] = 1;
  } else {
    slope = float(slope);
    attributes = {float_bits(float(slope))};
  }
  Tensor out = dispatch_tensor(op, x.shape(), {input}, attributes, (preserve || is_differentiable_float(x.dtype())) ? x.dtype() : DType::Float32),
         saved = out.detach();
  return tracked(out, {x}, [original = x, saved, op, slope](const Tensor& upstream) {
    const bool fused = op == "silu" || op == "gelu" || op == "leaky_relu";
    const bool low = is_low_precision_float(original.dtype());
    const Tensor x = low || (fused && is_narrow_float(original.dtype())) ? cast(original, DType::Float32) : original;
    const Tensor g = low || (fused && is_narrow_float(upstream.dtype())) ? cast(upstream, DType::Float32) : upstream;
    // Low precision extensions differentiate the FP32 operation before output quantization.
    const Tensor forward = low && (op == "sqrt" || op == "rsqrt" || op == "exp") ? unary(op, x, slope) : saved;
    Tensor d;
    if (op == "sqrt") {
      d = div(scalar(0.5f, x.backend()), forward);
    } else if (op == "rsqrt") {
      accumulate(original, mul(mul(g, -0.5f), pow(forward, 3)));
      return;
    } else if (op == "exp") {
      d = forward;
    } else if (op == "log") {
      d = div(scalar(1, x.backend()), x);
    } else if (op == "sin") {
      d = cos(x);
    } else if (op == "cos") {
      d = mul(sin(x), -1);
    } else if (op == "abs") {
      d = sign(x);
    } else if (op == "sign") {
      accumulate(original, Tensor::zeros(x.shape(), x.backend(), false, x.dtype()));
      return;
    } else if (op == "relu") {
      accumulate(original, where(le(x, scalar(0, x.backend())), scalar(0, x.backend()), g));
      return;
    } else if (op == "leaky_relu") {
      if (x.dtype() == DType::Float64) {
        d = where(gt(x, scalar(0, x.backend())), Tensor::from_double_data({}, {1.}, x.backend()),
                  Tensor::from_double_data({}, {double(slope)}, x.backend()));
      } else d = add(
          mul(cast(gt(x, scalar(0, x.backend())), DType::Float32), 1.f - float(slope)),
          slope);
    } else if (op == "silu") {
      Tensor sigmoid = div(scalar(1, x.backend()), add(exp(mul(x, -1)), 1));
      d = mul(sigmoid, add(mul(x, sub(scalar(1, x.backend()), sigmoid)), 1));
    } else if (op == "gelu") {
      if (x.dtype() == DType::Float64) {
        LocalFunction constant{[&](double value) { return Tensor::from_double_data({}, {value}, x.backend()); }};
        Tensor u = mul(add(x, mul(pow(x, 3), constant(.044715))), constant(.79788456080286535588));
        Tensor t = sub(div(constant(2), add(exp(mul(u, -2)), 1)), 1);
        d = add(mul(add(t, 1), .5f), mul(mul(mul(x, sub(constant(1), mul(t, t))),
                add(mul(mul(x, x), constant(.134145)), 1)), constant(.39894228040143267794)));
        accumulate(original, mul(g, d));
        return;
      }
      Tensor u = mul(add(x, mul(pow(x, 3), 0.044715f)), 0.7978845608028654f);
      Tensor t = sub(div(scalar(2, x.backend()), add(exp(mul(u, -2)), 1)), 1);
      d = add(mul(add(t, 1), 0.5f),
              mul(mul(mul(x, sub(scalar(1, x.backend()), mul(t, t))),
                      add(mul(mul(x, x), 0.134145f), 1)),
                  0.3989422804014327f));
    }
    accumulate(original, mul(g, d));
  });
}
Tensor sqrt(const Tensor& x) {
  return unary("sqrt", x);
}
Tensor rsqrt(const Tensor& x) {
  return unary("rsqrt", x);
}
Tensor exp(const Tensor& x) {
  return unary("exp", x);
}
Tensor log(const Tensor& x) {
  return unary("log", x);
}
Tensor sin(const Tensor& x) {
  return unary("sin", x);
}
Tensor cos(const Tensor& x) {
  return unary("cos", x);
}
Tensor abs(const Tensor& x) {
  return unary("abs", x);
}
Tensor sign(const Tensor& x) {
  return unary("sign", x);
}
Tensor relu(const Tensor& x) {
  return unary("relu", x);
}
Tensor silu(const Tensor& x) {
  return unary("silu", x);
}
Tensor gelu(const Tensor& x) {
  return unary("gelu", x);
}

Tensor swish(const Tensor& x) {
  return silu(x);
}
Tensor silu_mul(const Tensor& gate, const Tensor& up) {
  if (is_low_precision_float(gate.dtype()) || is_low_precision_float(up.dtype())) {
    require(gate.dtype() == up.dtype(), "Low precision SiLU multiplication requires matching dtypes");
    return tracked(dispatch_tensor("silu_mul", broadcast(gate.shape(), up.shape()), {gate, up}, {}, gate.dtype()), {gate, up}, [gate, up](const Tensor& gradient) {
      const Tensor x = cast(gate.detach(), DType::Float32), y = cast(up.detach(), DType::Float32), g = cast(gradient, DType::Float32);
      if (up.requires_grad()) accumulate(up, unbroadcast(mul(g, silu(x)), up.shape()));
      if (gate.requires_grad()) {
        const Tensor reduced = unbroadcast(mul(g, y), gate.shape());
        const Tensor sigmoid = div(scalar(1, gate.backend()), add(exp(mul(x, -1)), 1));
        const Tensor derivative = mul(sigmoid, add(mul(x, sub(scalar(1, gate.backend()), sigmoid)), 1));
        accumulate(gate, mul(reduced, derivative));
      }
    });
  }
  require(is_arithmetic_float(gate.dtype()) && gate.dtype() == up.dtype(),
          "silu_mul requires matching floating dtypes");
  Shape shape = broadcast(gate.shape(), up.shape());
  return tracked(dispatch_tensor("silu_mul", shape, {gate, up}, {}, gate.dtype()),
                 {gate, up}, [gate, up](const Tensor& g) {
    if (up.requires_grad()) accumulate(up, unbroadcast(mul(g, silu(gate)), up.shape()));
    if (gate.requires_grad()) {
      Tensor gradient = unbroadcast(mul(g, up), gate.shape());
      Tensor wide = cast(gate, gate.dtype() == DType::Float64 ? DType::Float64 : DType::Float32);
      Tensor sigmoid = div(scalar(1, gate.backend()), add(exp(mul(wide, -1)), 1));
      Tensor derivative = mul(sigmoid, add(mul(wide, sub(scalar(1, gate.backend()), sigmoid)), 1));
      accumulate(gate, mul(cast(gradient, wide.dtype()), derivative));
    }
  });
}
Tensor rotary_embedding(const Tensor& x, const Tensor& cosine, const Tensor& sine) {
  if (is_low_precision_float(x.dtype()) || is_low_precision_float(cosine.dtype()) || is_low_precision_float(sine.dtype())) {
    require(x.dtype() == cosine.dtype() && x.dtype() == sine.dtype(), "Low precision rotary operands must match");
    require(!x.shape().empty() && x.shape().back() % 2 == 0, "Rotary input requires an even trailing dimension");
    require(broadcast(x.shape(), cosine.shape()) == x.shape() && broadcast(x.shape(), sine.shape()) == x.shape(), "Rotary factors may not expand the input");
    return tracked(dispatch_tensor("rotary_embedding", x.shape(), {x, cosine, sine}, {}, x.dtype()), {x, cosine, sine}, [x, cosine, sine](const Tensor& gradient) {
      const Tensor value = cast(x.detach(), DType::Float32), c = cast(cosine.detach(), DType::Float32), s = cast(sine.detach(), DType::Float32), g = cast(gradient, DType::Float32);
      const int width = x.shape().back(), half = width / 2;
      if (x.requires_grad()) {
        const Tensor rotated = mul(g, s);
        const Tensor inverse = concat({slice(rotated, -1, half, width), mul(slice(rotated, -1, 0, half), -1)}, -1);
        accumulate(x, add(mul(g, c), inverse));
      }
      if (cosine.requires_grad()) accumulate(cosine, unbroadcast(mul(g, value), cosine.shape()));
      if (sine.requires_grad()) {
        const Tensor rotated = concat({mul(slice(value, -1, half, width), -1), slice(value, -1, 0, half)}, -1);
        accumulate(sine, unbroadcast(mul(g, rotated), sine.shape()));
      }
    });
  }
  require(is_arithmetic_float(x.dtype()) && cosine.dtype() == x.dtype() && sine.dtype() == x.dtype(),
          "rotary_embedding requires matching floating dtypes");
  require(!x.shape().empty() && x.shape().back() % 2 == 0,
          "rotary_embedding requires an even trailing dimension");
  require(broadcast(x.shape(), cosine.shape()) == x.shape() && broadcast(x.shape(), sine.shape()) == x.shape(),
          "rotary_embedding factors must broadcast to the input shape");
  return tracked(dispatch_tensor("rotary_embedding", x.shape(), {x, cosine, sine}, {}, x.dtype()),
                 {x, cosine, sine}, [x, cosine, sine](const Tensor& g) {
    int width = x.shape().back(), half = width / 2;
    if (x.requires_grad()) {
      Tensor rotated_gradient = mul(g, sine);
      Tensor inverse = concat({slice(rotated_gradient, -1, half, width),
                             mul(slice(rotated_gradient, -1, 0, half), -1)}, -1);
      accumulate(x, add(mul(g, cosine), inverse));
    }
    if (cosine.requires_grad()) accumulate(cosine, unbroadcast(mul(g, x), cosine.shape()));
    if (sine.requires_grad()) {
      Tensor rotated = concat({mul(slice(x, -1, half, width), -1), slice(x, -1, 0, half)}, -1);
      accumulate(sine, unbroadcast(mul(g, rotated), sine.shape()));
    }
  });
}
Tensor leaky_relu(const Tensor& x, double slope) {
  return unary("leaky_relu", x, slope);
}
Tensor reshape(const Tensor& x, Shape shape) {
  int inferred = -1;
  size_t known = 1;
  for (int i = 0; i < int(shape.size()); ++i) {
    if (shape[i] == -1) {
      require(inferred < 0, "Only one inferred reshape dimension is allowed");
      inferred = i;
    } else {
      require(shape[i] >= 0, "Invalid reshape dimension");
      known *= shape[i];
    }
  }
  if (inferred >= 0) {
    require(known != 0, "Cannot infer reshape dimension with zero known size");
    require(x.numel() % known == 0, "Cannot infer reshape dimension");
    shape[inferred] = int(x.numel() / known);
  }
  require(shape_element_count(shape) == x.numel(),
          "Reshape changes element count");
  std::shared_ptr<TensorNode> n = std::make_shared<TensorNode>();
  n->shape = shape;
  n->value = x.node()->value;
  n->operation = "reshape";
  return tracked(Tensor(n), {x}, [x](const Tensor& g) {
    accumulate(x, reshape(g, x.shape()));
  });
}
Tensor contiguous_view(const Tensor& x, Shape shape, size_t offset) {
  require(x.defined(), "Undefined contiguous view input");
  size_t count = shape_element_count(shape);
  std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>();
  node->shape = std::move(shape);
  node->value = backend_view(x.node()->value, offset, count);
  node->operation = "contiguous_view";
  return tracked(Tensor(node), {x}, [x, offset](const Tensor& g) {
    Tensor gradient = dispatch_tensor("slice_grad", {int(x.numel())}, {reshape(g, {int(g.numel())})},
                                    {0, int(offset), 1}, x.dtype());
    accumulate(x, reshape(gradient, x.shape()));
  });
}
Tensor permute(const Tensor& x, const std::vector<int>& axes) {
  require(axes.size() == x.shape().size(), "Permutation rank mismatch");
  Shape s(axes.size());
  std::vector<int> normalized, inverse(axes.size());
  std::unordered_set<int> seen;
  for (int i = 0; i < int(axes.size()); ++i) {
    int a = axis(axes[i], axes.size());
    require(seen.insert(a).second, "Repeated permutation axis");
    normalized.push_back(a);
    s[i] = x.shape()[a];
    inverse[a] = i;
  }
  return tracked(
      dispatch_tensor("permute", s, {x}, normalized, x.dtype()), {x},
      [x, inverse](const Tensor& g) { accumulate(x, permute(g, inverse)); });
}
Tensor transpose(const Tensor& x, int a, int b) {
  if (x.shape().empty()) {
    require((a == -1 || a == 0) && (b == -1 || b == 0), "Invalid scalar transpose axes");
    return reshape(x, {});
  }
  a = axis(a, x.shape().size());
  b = axis(b, x.shape().size());
  std::vector<int> p(x.shape().size());
  std::iota(p.begin(), p.end(), 0);
  std::swap(p[a], p[b]);
  return permute(x, p);
}
Tensor slice(const Tensor& x, int dim, int start, int stop, int step) {
  dim = axis(dim, x.shape().size());
  require(step > 0, "Slice step must be positive");
  int n = x.shape()[dim];
  if (start < 0) {
    start += n;
  }
  if (stop < 0) {
    stop += n;
  }
  start = std::max(0, std::min(n, start));
  stop = std::max(0, std::min(n, stop));
  Shape s = x.shape();
  s[dim] = stop <= start ? 0 : (stop - start + step - 1) / step;
  return tracked(
      dispatch_tensor("slice", s, {x}, {dim, start, step}, x.dtype()), {x},
      [x, dim, start, step](const Tensor& g) {
        accumulate(x, dispatch_tensor("slice_grad", x.shape(), {g},
                                      {dim, start, step}, (is_complex(x.dtype()) || x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32));
      });
}
static void validate_indices(const Tensor& ids, int count) {
  require(ids.dtype() == DType::Int32, "Indices must use int32 dtype");
  for (int32_t value : ids.int_data()) require(value >= 0 && value < count, "Index out of range");
}
Tensor argsort(const Tensor& x, int dim, bool descending) {
  require(is_arithmetic_float(x.dtype()) || is_low_precision_float(x.dtype()) || is_arithmetic_integer(x.dtype()), "Unsupported argsort dtype");
  if (x.shape().empty()) {
    require(dim == 0 || dim == -1, "Scalar argsort axis out of range");
    return reshape(argsort(reshape(x, {1}), 0, descending), {});
  }
  dim = axis(dim, x.shape().size());
  return dispatch_tensor("argsort", x.shape(), {x}, {dim, int(descending)}, DType::Int32);
}
std::pair<Tensor, Tensor> sort(const Tensor& x, int dim, bool descending) {
  if (x.shape().empty()) {
    require(dim == 0 || dim == -1, "Scalar sort axis out of range");
    std::pair<Tensor, Tensor> result = sort(reshape(x, {1}), 0, descending);
    return {reshape(result.first, {}), reshape(result.second, {})};
  }
  dim = axis(dim, x.shape().size());
  Tensor ids = argsort(x, dim, descending);
  // Stable indices and values retain the selected backend and original dtype.
  Tensor values = dispatch_tensor("gather", x.shape(), {x, ids}, {dim}, x.dtype());
  values = tracked(values, {x, ids}, [x, ids, dim](const Tensor& gradient) {
    accumulate(x, dispatch_tensor("sort_backward", x.shape(), {gradient, ids}, {dim},
                                  (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32));
  });
  return {values, ids};
}
Tensor gather(const Tensor& x, int dim, const Tensor& ids) {
  if ((is_complex(x.dtype()) || x.dtype() == DType::Float64 || is_low_precision_float(x.dtype()) || is_packed_integer(x.dtype())) && (x.shape().empty() || ids.shape().empty())) {
    dim = axis(dim, std::max(size_t(1), x.shape().size()));
    Tensor result = gather(x.shape().empty() ? reshape(x, {1}) : x, dim,
                         ids.shape().empty() ? reshape(ids, {1}) : ids);
    return ids.shape().empty() ? reshape(result, {}) : result;
  }
  dim = axis(dim, x.shape().size());
  require(ids.shape().size() == x.shape().size(), "Gather rank mismatch");
  for (int j = 0; j < int(x.shape().size()); ++j) {
    if (j != dim) {
      require(ids.shape()[j] <= x.shape()[j], "Gather dimension exceeds input");
    }
  }
  validate_indices(ids, x.shape()[dim]);
  return tracked(
      dispatch_tensor("gather", ids.shape(), {x, ids}, {dim}, x.dtype()), {x},
      [x, ids, dim](const Tensor& g) {
        if (is_complex(x.dtype())) {
          LocalFunction component{[&](const Tensor& value) {
            return dispatch_tensor("gather_grad", x.shape(), {value, ids}, {dim}, value.dtype());
          }};
          accumulate(x, complex(component(real(g)), component(imag(g))));
          return;
        }
        accumulate(x,
                   dispatch_tensor("gather_grad", x.shape(), {g, ids}, {dim},
                                   (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32));
      });
}
Tensor index_select(const Tensor& x, int dim, const Tensor& ids) {
  if ((is_complex(x.dtype()) || x.dtype() == DType::Float64 || is_low_precision_float(x.dtype()) || is_packed_integer(x.dtype())) && (x.shape().empty() || ids.shape().empty())) {
    dim = axis(dim, std::max(size_t(1), x.shape().size()));
    require(!x.shape().empty() || ids.numel() == 1, "Scalar selection requires exactly one index");
    Tensor result = index_select(x.shape().empty() ? reshape(x, {1}) : x, dim,
                               ids.shape().empty() ? reshape(ids, {1}) : ids);
    return x.shape().empty() ? reshape(result, {}) : result;
  }
  dim = axis(dim, x.shape().size());
  require(ids.shape().size() == 1, "index_select indices must be a vector");
  validate_indices(ids, x.shape()[dim]);
  Shape s = x.shape();
  s[dim] = int(ids.numel());
  return tracked(dispatch_tensor("index_select", s, {x, ids}, {dim}, x.dtype()),
                 {x}, [x, ids, dim](const Tensor& g) {
                   if (is_complex(x.dtype())) {
                     LocalFunction component{[&](const Tensor& value) {
                       return dispatch_tensor("index_select_grad", x.shape(), {value, ids}, {dim}, value.dtype());
                     }};
                     accumulate(x, complex(component(real(g)), component(imag(g))));
                     return;
                   }
                   accumulate(x, dispatch_tensor("index_select_grad", x.shape(),
                                                 {g, ids}, {dim}, (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32));
                 });
}
Tensor index_copy(const Tensor& x, int dim, const Tensor& ids, const Tensor& source) {
  dim = axis(dim, std::max(size_t(1), x.shape().size()));
  Tensor result = dispatch_tensor("index_copy", x.shape(), {x, ids, source}, {dim}, x.dtype());
  return tracked(result, {x, ids, source}, [x, ids, source, dim](const Tensor& gradient) {
    if (x.requires_grad())
      accumulate(x, index_copy(gradient, dim, ids, Tensor::zeros(source.shape(), source.backend(), false, source.dtype())));
    if (source.requires_grad()) {
      // The source derivative checks indices even for empty slices,
      // although the forward and destination derivative do not inspect them.
      MemoryReservation host(Backend::CPU, checked_memory_add(ids.nbytes(), checked_memory_mul(ids.numel(), sizeof(int32_t))));
      std::vector<uint8_t> bytes = ids.raw_data();
      std::vector<int32_t> values(ids.numel());
      int length = x.shape().empty() ? 1 : x.shape()[dim];
      for (size_t i = 0; i < values.size(); ++i) {
        int64_t value;
        if (ids.dtype() == DType::Int64) std::memcpy(&value, bytes.data() + i * 8, 8);
        else { int32_t narrow; std::memcpy(&narrow, bytes.data() + i * 4, 4); value = narrow; }
        require(value >= 0 && value < length, "Index copy source gradient index out of range");
        values[i] = int32_t(value);
      }
      Tensor gradient_ids = Tensor::from_int_data({int(values.size())}, values, x.backend());
      Tensor selected = source.numel() ? index_select(gradient.shape().empty() ? contiguous_view(gradient, {1}) : gradient,
                                                    dim, gradient_ids) : Tensor::zeros(source.shape(), source.backend(), false, source.dtype());
      accumulate(source, contiguous_view(selected, source.shape()));
    }
  });
}
void Tensor::assign_indexed(int dim, const Tensor& ids, const Tensor& source) {
  require(!requires_grad() && !source.requires_grad(), "Indexed mutation requires tensors without gradients");
  dim = axis(dim, std::max(size_t(1), shape().size()));
  dispatch_tensor_into("index_copy", *this, {*this, ids, source}, {dim});
  ++node()->value->version;
}
Tensor scatter(const Tensor& x, int dim, const Tensor& ids,
               const Tensor& source) {
  if (is_complex(x.dtype()) || is_complex(source.dtype())) {
    require(x.dtype() == source.dtype(), "Complex scatter dtypes must match");
    return complex(scatter(real(x), dim, ids, real(source)), scatter(imag(x), dim, ids, imag(source)));
  }
  require(x.dtype() == source.dtype(), "Scatter source and destination dtypes must match");
  if ((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype()) || is_packed_integer(x.dtype())) && (x.shape().empty() || ids.shape().empty())) {
    dim = axis(dim, std::max(size_t(1), x.shape().size()));
    require(source.shape() == ids.shape(), "Scatter source/index shape mismatch");
    Tensor result = scatter(x.shape().empty() ? reshape(x, {1}) : x, dim,
                          ids.shape().empty() ? reshape(ids, {1}) : ids,
                          source.shape().empty() ? reshape(source, {1}) : source);
    return x.shape().empty() ? reshape(result, {}) : result;
  }
  dim = axis(dim, x.shape().size());
  require(
      source.shape() == ids.shape() && ids.shape().size() == x.shape().size(),
      "Scatter source/index shape mismatch");
  for (int d = 0; d < int(x.shape().size()); ++d) {
    if (d != dim) {
      require(ids.shape()[d] <= x.shape()[d],
              "Scatter shape exceeds destination");
    }
  }
  validate_indices(ids, x.shape()[dim]);
  return tracked(
      dispatch_tensor("scatter", x.shape(), {x, ids, source}, {dim}, x.dtype()),
      {x, source}, [x, ids, source, dim](const Tensor& g) {
        if (x.requires_grad()) {
          accumulate(
              x, dispatch_tensor(
                     "scatter", x.shape(),
                     {g, ids, Tensor::zeros(ids.shape(), x.backend(), false, (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32)},
                     {dim}, (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32));
        }
        if (source.requires_grad()) {
          accumulate(source, dispatch_tensor("scatter_source_grad",
                                             source.shape(), {g, ids}, {dim}, (source.dtype() == DType::Float64 || is_low_precision_float(source.dtype())) ? source.dtype() : DType::Float32));
        }
      });
}
Tensor concat(const std::vector<Tensor>& xs, int dim) {
  require(!xs.empty(), "concat requires inputs");
  Tensor result = xs[0];
  dim = axis(dim, result.shape().size());
  for (size_t k = 1; k < xs.size(); ++k) {
    Tensor a = result, b = xs[k];
    require(a.shape().size() == b.shape().size(), "concat rank mismatch");
    require(a.dtype() == b.dtype(), "concat dtype mismatch");
    Shape s = a.shape();
    for (int d = 0; d < int(s.size()); ++d) {
      if (d != dim) {
        require(s[d] == b.shape()[d], "concat dimension mismatch");
      }
    }
    int boundary = s[dim];
    s[dim] += b.shape()[dim];
    result = tracked(dispatch_tensor("concat", s, {a, b}, {dim}, a.dtype()),
                     {a, b}, [a, b, dim, boundary](const Tensor& g) {
                       accumulate(a, slice(g, dim, 0, boundary));
                       accumulate(b, slice(g, dim, boundary, g.shape()[dim]));
                     });
  }
  return result;
}
std::vector<Tensor> split(const Tensor& x, int size, int dim) {
  dim = axis(dim, x.shape().size());
  require(size > 0, "split size must be positive");
  std::vector<Tensor> out;
  for (int start = 0; start < x.shape()[dim]; start += size) {
    out.push_back(slice(x, dim, start, std::min(start + size, x.shape()[dim])));
  }
  return out;
}
Tensor repeat(const Tensor& x, const Shape& reps) {
  require(reps.size() >= x.shape().size(),
          "repeat rank is smaller than input rank");
  Shape padded(reps.size() - x.shape().size(), 1);
  padded.insert(padded.end(), x.shape().begin(), x.shape().end());
  Shape s = padded;
  for (size_t i = 0; i < reps.size(); ++i) {
    require(reps[i] >= 0, "repeat factors must be nonnegative");
    s[i] *= reps[i];
  }
  Tensor y = reshape(x, padded);
  return tracked(
      dispatch_tensor("repeat", s, {y}, {}, x.dtype()), {x},
      [x, padded](const Tensor& g) {
        if (is_complex(x.dtype())) {
          LocalFunction component{[&](const Tensor& value) {
            return reshape(dispatch_tensor("repeat_grad", padded, {value}, {}, value.dtype()), x.shape());
          }};
          accumulate(x, complex(component(real(g)), component(imag(g))));
          return;
        }
        accumulate(
            x, reshape(dispatch_tensor("repeat_grad", padded, {g}, {},
                                       (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32), x.shape()));
      });
}
Tensor expand(const Tensor& x, const Shape& shape) {
  require(shape.size() >= x.shape().size(), "expand cannot remove dimensions");
  Shape s = shape;
  for (size_t i = 0; i < s.size(); ++i) {
    int j = int(i) - int(s.size() - x.shape().size());
    if (s[i] == -1) {
      require(j >= 0, "Cannot infer a new expand dimension");
      s[i] = x.shape()[j];
    }
  }
  require(broadcast(x.shape(), s) == s, "Invalid expand target");
  return tracked(
      dispatch_tensor("expand", s, {x}, {}, x.dtype()), {x},
      [x](const Tensor& g) {
        const Tensor values = is_low_precision_float(g.dtype()) ? cast(g, DType::Float32) : g;
        accumulate(x, unbroadcast(values, x.shape()));
      });
}
Tensor squeeze(const Tensor& x, int dim) {
  if (x.shape().empty()) {
    require(dim == -1 || dim == 0, "Scalar squeeze axis out of range");
    return reshape(x, {});
  }
  dim = axis(dim, x.shape().size());
  Shape s = x.shape();
  if (s[dim] == 1) {
    s.erase(s.begin() + dim);
  }
  return reshape(x, s);
}
Tensor squeeze(const Tensor& x) {
  Shape s;
  for (int d : x.shape()) {
    if (d != 1) {
      s.push_back(d);
    }
  }
  return reshape(x, s);
}
Tensor unsqueeze(const Tensor& x, int dim) {
  if (dim < 0) {
    dim += int(x.shape().size()) + 1;
  }
  require(dim >= 0 && dim <= int(x.shape().size()),
          "unsqueeze axis out of range");
  Shape s = x.shape();
  s.insert(s.begin() + dim, 1);
  return reshape(x, s);
}
static Tensor reduce(const std::string& op, const Tensor& x, int dim,
                     bool keepdim, bool all = false) {
  if (is_complex(x.dtype())) return complex_reduce(op, x, dim, keepdim, all);
  const bool scalar_input = x.shape().empty();
  require(!scalar_input || all || dim == 0 || dim == -1, "Scalar reduction axis out of range");
  int d = all || scalar_input ? -1 : axis(dim, x.shape().size());
  Shape s = x.shape();
  if (all || scalar_input) {
    s = keepdim ? Shape(s.size(), 1) : Shape{};
  } else if (keepdim) {
    s[d] = 1;
  } else {
    s.erase(s.begin() + d);
  }
  require(op != "mean" || is_differentiable_float(x.dtype()), "Mean requires floating input");
  if (op == "max" || op == "min" || op == "amax" || op == "amin") {
    require(d < 0 ? x.numel() > 0 : x.shape()[d] > 0, "Extremum reduction requires a nonempty reduction axis");
  }
  DType dtype = x.dtype();
  if ((op == "sum" || op == "prod") && is_arithmetic_integer(dtype)) dtype = DType::Int64;
  Tensor out = dispatch_tensor(op, s, {x}, {d, int(keepdim)}, dtype),
         saved = out.detach();
  int kind = op == "mean"                    ? 1
             : op == "amax"                  ? 2
             : op == "amin"                  ? 3
             : op == "max"                   ? 5
             : op == "min"                   ? 6
             : op == "prod"                  ? 4
                                             : 0;
  return tracked(out, {x}, [x, saved, d, keepdim, kind](const Tensor& g) {
    accumulate(x, dispatch_tensor("reduce_grad", x.shape(), {x, saved, g},
                                  {d, int(keepdim), kind}, x.dtype() == DType::Float64 ? DType::Float64 : DType::Float32));
  });
}
Tensor cumsum(const Tensor& x, int dim) {
  if (is_complex(x.dtype())) return complex(cumsum(real(x), dim), cumsum(imag(x), dim));
  require(is_differentiable_float(x.dtype()) || is_arithmetic_integer(x.dtype()),
          "Cumsum arithmetic is not implemented for this dtype");
  if (x.shape().empty()) {
    require(dim == 0 || dim == -1, "Scalar cumsum axis out of range");
    dim = 0;
  } else {
    dim = axis(dim, x.shape().size());
  }
  DType dtype = is_differentiable_float(x.dtype()) ? x.dtype() : DType::Int64;
  Tensor result = dispatch_tensor("cumsum", x.shape(), {x}, {dim}, dtype);
  return tracked(result, {x}, [x, dim](const Tensor& gradient) {
    accumulate(x, dispatch_tensor("cumsum_backward", x.shape(), {gradient}, {dim}, gradient.dtype()));
  });
}
Tensor sum(const Tensor& x, int dimension, bool keepdim) {
  return reduce("sum", x, dimension, keepdim);
}
Tensor mean(const Tensor& x, int dimension, bool keepdim) {
  return reduce("mean", x, dimension, keepdim);
}
Tensor max(const Tensor& x, int dimension, bool keepdim) {
  return reduce("max", x, dimension, keepdim);
}
Tensor min(const Tensor& x, int dimension, bool keepdim) {
  return reduce("min", x, dimension, keepdim);
}
Tensor amax(const Tensor& x, int dimension, bool keepdim) {
  return reduce("amax", x, dimension, keepdim);
}
Tensor amin(const Tensor& x, int dimension, bool keepdim) {
  return reduce("amin", x, dimension, keepdim);
}
Tensor prod(const Tensor& x, int dimension, bool keepdim) {
  return reduce("prod", x, dimension, keepdim);
}

Tensor sum_all(const Tensor& x) {
  return reduce("sum", x, 0, false, true);
}
Tensor mean_all(const Tensor& x) {
  return reduce("mean", x, 0, false, true);
}
static std::vector<int> random_attributes(DType dtype, uint32_t seed, double first, double second, bool dropout = false) {
  std::vector<int> attributes{int(seed), 0, float_bits(float(first)), dropout ? 1 : float_bits(float(second))};
  if (dtype == DType::Float64 || is_low_precision_float(dtype) || is_compute_float(dtype)) {
    attributes.resize(9);
    std::memcpy(attributes.data() + 4, &first, sizeof(first));
    std::memcpy(attributes.data() + 6, &second, sizeof(second));
    attributes[8] = 1;
  }
  return attributes;
}
Tensor Tensor::uniform(Shape shape, double low, double high, uint32_t seed,
                       Backend b, bool grad, DType dtype) {
  if (is_complex(dtype)) {
    const size_t count = shape_element_count(shape);
    require(count <= size_t(INT32_MAX) / 2, "Complex random component count exceeds indexing limit");
    Tensor components = Tensor::uniform({int(count), 2}, low, high, seed, b, false, complex_component_dtype(dtype));
    Tensor result = reshape(complex(squeeze(slice(components, 1, 0, 1), 1),
                                  squeeze(slice(components, 1, 1, 2), 1)), shape).detach();
    result.node()->requires_grad = grad;
    return result;
  }
  require(is_arithmetic_float(dtype) || is_low_precision_float(dtype), "Uniform sampling requires an implemented floating dtype");
  require(std::isfinite(low) && std::isfinite(high) && low <= high &&
          (dtype != DType::Float64 || std::isfinite(high - low)), "Invalid uniform range");
  Tensor t = dispatch_tensor("uniform", shape, {}, random_attributes(dtype, seed, low, high), dtype, b);
  t.node()->requires_grad = grad;
  return t;
}
Tensor Tensor::normal(Shape shape, double mean, double sd, uint32_t seed,
                      Backend b, bool grad, DType dtype) {
  if (is_complex(dtype)) {
    const size_t count = shape_element_count(shape);
    require(count <= size_t(INT32_MAX) / 2, "Complex random component count exceeds indexing limit");
    Tensor components = Tensor::normal({int(count), 2}, mean, sd / std::sqrt(2.), seed, b, false, complex_component_dtype(dtype));
    Tensor result = reshape(complex(squeeze(slice(components, 1, 0, 1), 1),
                                  squeeze(slice(components, 1, 1, 2), 1)), shape).detach();
    result.node()->requires_grad = grad;
    return result;
  }
  require(is_arithmetic_float(dtype) || is_low_precision_float(dtype), "Normal sampling requires an implemented floating dtype");
  require(sd >= 0, "Invalid normal parameters");
  Tensor t = dispatch_tensor("normal", shape, {}, random_attributes(dtype, seed, mean, sd), dtype, b);
  t.node()->requires_grad = grad;
  return t;
}
Tensor dropout(const Tensor& x, double p, uint32_t seed, bool training) {
  require(p >= 0 && p <= 1, "Dropout probability must be in [0,1]");
  if (!training || p == 0 || x.numel() == 0) return x;
  require(is_arithmetic_float(x.dtype()) || is_low_precision_float(x.dtype()), "Training dropout requires floating input");
  const std::vector<int> attributes = random_attributes(x.dtype(), seed, p, 0., true);
  return tracked(dispatch_tensor("dropout", x.shape(), {x}, attributes, x.dtype()),
                 {x}, [x, attributes](const Tensor& g) {
                   std::vector<int> backward = attributes;
                   if (x.dtype() != DType::Float64) { backward.resize(10); backward[operator_attr::dropout::kBackward] = 1; }
                   accumulate(x, dispatch_tensor("dropout", g.shape(), {g}, backward, x.dtype()));
                 });
}
static Tensor matrix_backward_product(const Tensor& a, const Tensor& b, bool low, bool folded = true, bool vector_derivative = false) {
  // A one-term contraction is an outer product. Preserve its zero signs before
  // quantizing the leaf gradient, as the FP32 BLAS path does.
  if (low && a.shape().back() == 1 && b.shape()[b.shape().size() - 2] == 1) {
    Tensor product = mul(a, b);
    const bool gemv = !vector_derivative && (a.shape()[a.shape().size() - 2] == 1 || b.shape().back() == 1);
    return a.backend() == Backend::CPU && (!folded || gemv) ? add(product, 0.) : product;
  }
  return matmul(a, b);
}
static Tensor matmul_impl(const Tensor& a, const Tensor& b, bool vector_left = false, bool vector_right = false) {
  require(a.dtype() == b.dtype(), "matmul requires matching dtypes");
  require(!a.shape().empty() && !b.shape().empty(),
          "matmul requires rank >= 1");
  if (a.shape().size() == 1 || b.shape().size() == 1) {
    bool av = a.shape().size() == 1, bv = b.shape().size() == 1;
    require(a.dtype() != DType::Bool || !av || !bv, "Bool dot products are not supported");
    Tensor y = matmul_impl(av ? unsqueeze(a, 0) : a, bv ? unsqueeze(b, -1) : b, av, bv);
    if (av) {
      y = squeeze(y, int(y.shape().size()) - 2);
    }
    if (bv) {
      y = squeeze(y, -1);
    }
    return y;
  }
  int ar = int(a.shape().size()), br = int(b.shape().size());
  require(a.shape()[ar - 1] == b.shape()[br - 2],
          "matmul inner dimension mismatch");
  Shape s = broadcast(Shape(a.shape().begin(), a.shape().end() - 2),
                      Shape(b.shape().begin(), b.shape().end() - 2));
  s.push_back(a.shape()[ar - 2]);
  s.push_back(b.shape()[br - 1]);
  if (a.dtype() == DType::Bool) {
    require(a.backend() == b.backend(), "Mixed-backend operation");
    require(a.shape().back() == 0 || shape_element_count(s) == 0, "Nonempty Bool matrix products are not supported");
    return Tensor::zeros(s, a.backend(), false, DType::Bool);
  }
  const bool folded = (ar <= 2 && br <= 2) || (ar >= 3 && br <= 2) ||
      (ar <= 2 && br >= 3 && (a.requires_grad() || (vector_left && a.shape().back() == 1)));
  return tracked(
      dispatch_tensor("matmul", s, {a, b}, is_low_precision_float(a.dtype()) ? std::vector<int>{0, int(folded)} : std::vector<int>{}, a.dtype()), {a, b}, [left = a, right = b, folded, vector_left, vector_right](const Tensor& upstream) {
        const bool low = is_low_precision_float(left.dtype());
        const Tensor a = low ? cast(left, DType::Float32) : left;
        const Tensor b = low ? cast(right, DType::Float32) : right;
        const Tensor g = low ? cast(upstream, DType::Float32) : upstream;
        if (left.requires_grad()) {
          // Retain the transposed weight layout so native CPU GEMM follows the
          // same dot reduction as the forward transposed-matrix primitive.
          Tensor gradient = !low && left.dtype() == DType::Float32 && left.backend() == Backend::CPU ?
              matmul_transposed(g, b) : matrix_backward_product(g, transpose(b, -2, -1), low, folded, vector_left || vector_right);
          accumulate(left, unbroadcast(gradient, a.shape()));
        }
        if (right.requires_grad()) accumulate(right, unbroadcast(matrix_backward_product(transpose(a, -2, -1), g, low, folded, vector_left || vector_right), b.shape()));
      });
}
Tensor matmul(const Tensor& a, const Tensor& b) {
  if (const DType target = autocast::dtype(); autocast::should_cast({&a, &b}, target))
    return matmul(cast(a, target), cast(b, target));
  if (is_complex(a.dtype()) || is_complex(b.dtype())) return complex_matrix(a, b);
  return matmul_impl(a, b);
}
Tensor bmm(const Tensor& a, const Tensor& b) {
  require(a.shape().size() == 3 && b.shape().size() == 3 &&
              a.shape()[0] == b.shape()[0],
          "bmm requires equal rank-3 batch sizes");
  return matmul(a, b);
}
Tensor matmul_transposed(const Tensor& a, const Tensor& b) {
  if (const DType target = autocast::dtype(); autocast::should_cast({&a, &b}, target))
    return matmul_transposed(cast(a, target), cast(b, target));
  require(a.shape().size() >= 2 && b.shape().size() >= 2,
          "Transposed matmul requires rank >= 2");
  require(a.dtype() == b.dtype(), "Transposed matmul requires matching dtypes");
  require(a.shape().back() == b.shape().back(), "Transposed matmul inner dimension mismatch");
  if (is_complex(a.dtype())) return complex_matrix(a, transpose(b, -2, -1));
  Shape shape = broadcast(Shape(a.shape().begin(), a.shape().end() - 2),
                          Shape(b.shape().begin(), b.shape().end() - 2));
  shape.push_back(a.shape()[a.shape().size() - 2]);
  shape.push_back(b.shape()[b.shape().size() - 2]);
  if (a.dtype() == DType::Bool) {
    require(a.backend() == b.backend(), "Mixed-backend operation");
    require(a.shape().back() == 0 || shape_element_count(shape) == 0, "Nonempty Bool matrix products are not supported");
    return Tensor::zeros(shape, a.backend(), false, DType::Bool);
  }
  const bool folded = (a.shape().size() <= 2 && b.shape().size() <= 2) ||
      (a.shape().size() >= 3 && b.shape().size() <= 2) ||
      (a.shape().size() <= 2 && b.shape().size() >= 3 && a.requires_grad());
  return tracked(dispatch_tensor("matmul", shape, {a, b}, is_low_precision_float(a.dtype()) ? std::vector<int>{1, int(folded)} : std::vector<int>{1}, a.dtype()), {a, b},
                 [left = a, right = b, folded](const Tensor& gradient) {
    const bool low = is_low_precision_float(left.dtype());
    const Tensor a = low ? cast(left, DType::Float32) : left;
    const Tensor b = low ? cast(right, DType::Float32) : right;
    const Tensor g = low ? cast(gradient, DType::Float32) : gradient;
    if (left.requires_grad()) accumulate(left, unbroadcast(matrix_backward_product(g, b, low, folded), a.shape()));
    if (right.requires_grad()) accumulate(right, unbroadcast(matrix_backward_product(transpose(g, -2, -1), a, low, folded), b.shape()));
  });
}
Tensor addmm(const Tensor& c, const Tensor& a, const Tensor& b, double beta,
             double alpha) {
  require(a.dtype() == b.dtype() && a.dtype() == c.dtype(), "addmm requires matching dtypes");
  require(a.shape().size() == 2 && b.shape().size() == 2 && a.shape()[1] == b.shape()[0], "addmm matrix dimensions mismatch");
  Shape shape{a.shape()[0], b.shape()[1]};
  require(broadcast(c.shape(), shape) == shape, "addmm addend shape mismatch");
  if (is_complex(a.dtype())) {
    require(c.backend() == a.backend() && b.backend() == a.backend(), "Mixed-backend complex addmm");
    if (a.dtype() == DType::Complex32)
      return cast(addmm(cast(c, DType::Complex64), cast(a, DType::Complex64), cast(b, DType::Complex64), beta, alpha), a.dtype());
    Tensor product = mul(complex_matrix(a, b), alpha);
    if (beta != 0) return add(product, mul(c, beta));
    // Preserve the zero-gradient edge without reading a possibly nonfinite addend.
    return graph::tracked(product.detach(), {product, c}, [product, c](const Tensor& gradient) {
      if (product.requires_grad()) accumulate(product, gradient);
      if (c.requires_grad()) accumulate(c, Tensor::zeros(c.shape(), c.backend(), false, c.dtype()));
    });
  }
  std::vector<int> attributes;
  if (a.dtype() == DType::Float64) {
    // Double coefficients occupy two descriptor words each; other dtypes keep
    // the established Float32 scalar conversion and raw kernel ABI.
    attributes.resize(4);
    std::memcpy(attributes.data(), &alpha, sizeof(alpha));
    std::memcpy(attributes.data() + 2, &beta, sizeof(beta));
  } else {
    alpha = float(alpha);
    beta = float(beta);
    attributes = {float_bits(float(alpha)), float_bits(float(beta))};
  }
  if (a.dtype() == DType::Bool) {
    require(a.backend() == b.backend() && a.backend() == c.backend(), "Mixed-backend operation");
    if (shape_element_count(shape) == 0) return Tensor::zeros(shape, a.backend(), false, DType::Bool);
    require(a.shape()[1] == 0, "Nonempty Bool addmm products are not supported");
    if (beta == 0) return Tensor::zeros(shape, a.backend(), false, DType::Bool);
    // The Float32 scalar ABI cannot be multiplied into Bool output on CPU.
    require(a.backend() == Backend::CUDA, "CPU Bool addmm requires zero beta for a zero-width product");
    return expand(logical_or(c, c), shape);
  }
  if (!is_differentiable_float(a.dtype())) {
    const DType coefficient_dtype = is_packed_integer(a.dtype()) ? DType::Int32 : a.dtype();
    validate_integer_matrix_coefficient(float(alpha), coefficient_dtype);
    validate_integer_matrix_coefficient(float(beta), coefficient_dtype);
    return dispatch_tensor("addmm", shape, {a, b, c}, attributes, a.dtype());
  }
  return tracked(dispatch_tensor("addmm", shape, {a, b, c}, attributes, a.dtype()),
                 {c, a, b}, [c, left = a, right = b, alpha, beta](const Tensor& upstream) {
                   const bool low = is_low_precision_float(left.dtype());
                   const Tensor a = low ? cast(left, DType::Float32) : left;
                   const Tensor b = low ? cast(right, DType::Float32) : right;
                   const Tensor g = low ? cast(upstream, DType::Float32) : upstream;
                   if (c.requires_grad()) accumulate(c, unbroadcast(mul(g, beta), c.shape()));
                   if (left.requires_grad()) accumulate(left, mul(matrix_backward_product(g, transpose(b, 0, 1), low), alpha));
                   if (right.requires_grad()) accumulate(right, mul(matrix_backward_product(transpose(a, 0, 1), g, low), alpha));
                 });
}
Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) {
  if (const DType target = autocast::dtype(); autocast::should_cast({&x, &w, &b}, target))
    return linear(cast(x, target), cast(w, target), autocast::cast_optional(b, target));
  require(!x.shape().empty() && w.shape().size() == 2 && x.shape().back() == w.shape()[1],
          "Linear expects input [...,in] and weight [out,in]");
  require(x.dtype() == w.dtype() &&
              (!b.defined() || b.dtype() == x.dtype()), "Linear requires matching numeric dtypes");
  require(!b.defined() || b.shape().empty() || b.shape() == Shape{w.shape()[0]}, "Linear bias must be scalar or match output features");
  if (is_complex(x.dtype())) {
    if (x.dtype() == DType::Complex32)
      return cast(linear(cast(x, DType::Complex64), cast(w, DType::Complex64),
                         b.defined() ? cast(b, DType::Complex64) : Tensor{}), x.dtype());
    Tensor result = matmul(x, transpose(w, 0, 1));
    return b.defined() ? add(result, b) : result;
  }
  Shape shape = x.shape();
  shape.back() = w.shape()[0];
  if (x.dtype() == DType::Bool) {
    require(x.backend() == w.backend() && (!b.defined() || b.backend() == x.backend()), "Mixed-backend operation");
    if (shape_element_count(shape) == 0) return Tensor::zeros(shape, x.backend(), false, DType::Bool);
    require(x.shape().back() == 0, "Nonempty Bool linear products are not supported");
    if (!b.defined()) return Tensor::zeros(shape, x.backend(), false, DType::Bool);
    // Contiguous rank-two/three and vector-bias CPU paths use addmm; the
    // remaining scalar-bias paths add Bool values after the empty product.
    const bool separate_bias = b.shape().empty() && x.shape().size() != 2 && x.shape().size() != 3;
    require(x.backend() == Backend::CUDA || separate_bias, "CPU Bool linear bias is unsupported for this shape");
    return expand(logical_or(b, b), shape);
  }
  return tracked(dispatch_tensor("linear", shape, {x, w, b}, {}, x.dtype()), {x, w, b},
                 [x, w, b](const Tensor& g) {
    size_t rows = 1;
    for (size_t axis = 0; axis + 1 < x.shape().size(); ++axis) rows *= size_t(x.shape()[axis]);
    const bool low = is_low_precision_float(x.dtype());
    const Tensor matrix = reshape(low ? cast(x, DType::Float32) : x, {int(rows), w.shape()[1]});
    const Tensor upstream = reshape(low ? cast(g, DType::Float32) : g, {int(rows), w.shape()[0]});
    if (x.requires_grad()) accumulate(x, reshape(matrix_backward_product(upstream, low ? cast(w, DType::Float32) : w, low), x.shape()));
    if (w.requires_grad()) accumulate(w, matrix_backward_product(transpose(upstream, 0, 1), matrix, low));
    if (b.requires_grad()) accumulate(b, b.shape().empty() ? sum_all(upstream) : sum(upstream, 0));
  });
}
Tensor packed_linear(const Tensor& x, const Tensor& w, const std::vector<int>& widths, const Tensor& b) {
  require(!x.shape().empty() && w.shape().size() == 2 && x.shape().back() == w.shape()[1],
          "Packed linear expects input [...,in] and weight [sum(out),in]");
  require(x.dtype() == w.dtype() && (is_differentiable_float(x.dtype()) || is_complex(x.dtype()) || is_arithmetic_integer(x.dtype())) &&
          (!b.defined() || b.dtype() == x.dtype()), "Packed linear requires matching numeric dtypes");
  require(!widths.empty() && widths.size() < size_t(KernelParams::kAttributeCount), "Invalid packed projection count");
  int columns = 0;
  for (int width : widths) {
    require(width >= 0 && width <= w.shape()[0] - columns, "Invalid packed projection width");
    columns += width;
  }
  require(columns == w.shape()[0], "Packed projection widths differ from weight shape");
  require(!b.defined() || b.shape() == Shape{columns}, "Packed linear bias must match all output features");
  size_t rows = shape_element_count(Shape(x.shape().begin(), x.shape().end() - 1));
  size_t count = shape_element_count({int(rows), columns});
  if (is_complex(x.dtype())) {
    if (x.dtype() == DType::Complex32)
      return cast(packed_linear(cast(x, DType::Complex64), cast(w, DType::Complex64), widths,
                                b.defined() ? cast(b, DType::Complex64) : Tensor{}), x.dtype());
    std::vector<Tensor> projections;
    int begin = 0;
    for (int width : widths) {
      const Tensor weight = slice(w, 0, begin, begin + width);
      const Tensor bias = b.defined() ? slice(b, 0, begin, begin + width) : Tensor{};
      projections.push_back(reshape(linear(x, weight, bias), {int(rows) * width}));
      begin += width;
    }
    return concat(projections, 0);
  }
  if (x.dtype() == DType::Bool) {
    require(x.backend() == w.backend() && (!b.defined() || b.backend() == x.backend()), "Mixed-backend operation");
    if (!count) return Tensor::zeros({0}, x.backend(), false, DType::Bool);
    require(x.shape().back() == 0, "Nonempty Bool packed products are not supported");
    if (!b.defined()) return Tensor::zeros({int(count)}, x.backend(), false, DType::Bool);
    require(x.backend() == Backend::CUDA, "CPU Bool packed linear bias is unsupported for a zero-width product");
    Tensor bias = logical_or(b, b);
    std::vector<Tensor> projections;
    int begin = 0;
    for (int width : widths) {
      Shape shape = x.shape();
      shape.back() = width;
      projections.push_back(reshape(expand(slice(bias, 0, begin, begin + width), shape), {int(rows) * width}));
      begin += width;
    }
    return concat(projections, 0);
  }
  std::vector<int> attributes{int(widths.size())};
  attributes.insert(attributes.end(), widths.begin(), widths.end());
  return tracked(dispatch_tensor("packed_linear", {int(count)}, {x, w, b}, attributes, x.dtype()), {x, w, b},
                 [x, w, b, widths, rows, columns](const Tensor& g) {
    const bool low = is_low_precision_float(x.dtype());
    const Tensor values = low ? cast(x, DType::Float32) : x;
    const Tensor weights = low ? cast(w, DType::Float32) : w;
    const Tensor gradient = low ? cast(g, DType::Float32) : g;
    Tensor matrix = reshape(values, {int(rows), w.shape()[1]});
    Tensor input_gradient;
    std::vector<Tensor> weight_gradients(widths.size()), bias_gradients(widths.size());
    int end = columns;
    for (size_t index = widths.size(); index-- > 0;) {
      int width = widths[index], begin = end - width;
      Tensor upstream = contiguous_view(gradient, {int(rows), width}, rows * size_t(begin));
      if (x.requires_grad()) {
        Tensor weight = contiguous_view(weights, {width, w.shape()[1]}, size_t(begin) * w.shape()[1]);
        Tensor piece = reshape(matrix_backward_product(upstream, weight, low), x.shape());
        if (low) input_gradient = input_gradient.defined() ? add(input_gradient, piece) : piece;
        else accumulate(x, piece);
      }
      if (w.requires_grad()) weight_gradients[index] = matrix_backward_product(transpose(upstream, 0, 1), matrix, low);
      if (b.requires_grad()) bias_gradients[index] = sum(upstream, 0);
      end = begin;
    }
    if (low && x.requires_grad()) accumulate(x, input_gradient);
    if (w.requires_grad()) {
      Tensor gradient = concat(weight_gradients, 0);
      // Slice backward adds each projection into a zero-filled full weight.
      if (low && widths.size() > 1) gradient = add(gradient, 0.);
      accumulate(w, gradient);
    }
    if (b.requires_grad()) accumulate(b, concat(bias_gradients, 0));
  });
}
static Tensor triangular(const std::string& op, const Tensor& x, int diagonal) {
  require(x.shape().size() >= 2, "Triangular operation requires rank >=2");
  return tracked(dispatch_tensor(op, x.shape(), {x}, {diagonal}, x.dtype()),
                 {x}, [x, diagonal, op](const Tensor& g) {
                   accumulate(x,
                              dispatch_tensor(op, g.shape(), {g}, {diagonal}, g.dtype()));
                 });
}
Tensor triu(const Tensor& x, int d) {
  return triangular("triu", x, d);
}
Tensor tril(const Tensor& x, int d) {
  return triangular("tril", x, d);
}
Tensor softmax(const Tensor& x, int dim) {
  if (autocast::should_cast({&x}, DType::Float32)) return softmax(cast(x, DType::Float32), dim);
  require(is_differentiable_float(x.dtype()), "Softmax requires floating input");
  if (x.shape().empty()) {
    require(dim == 0 || dim == -1, "Scalar softmax axis out of range");
    return reshape(softmax(reshape(x, {1}), 0), {});
  }
  dim = axis(dim, x.shape().size());
  if (is_low_precision_float(x.dtype())) {
    Tensor saved = dispatch_tensor("softmax", x.shape(), {x}, {dim, 1}, DType::Float32);
    Tensor y = cast(saved, x.dtype());
    return tracked(y, {x}, [x, saved, dim](const Tensor& g) {
      accumulate(x, dispatch_tensor("softmax", x.shape(), {saved, g}, {dim, 2}, x.dtype()));
    });
  }
  Tensor y = dispatch_tensor("softmax", x.shape(), {x}, {dim}, x.dtype()),
         saved = y.detach();
  return tracked(y, {x}, [x, saved, dim](const Tensor& g) {
    const Tensor values = is_narrow_float(saved.dtype()) ? cast(saved, DType::Float32) : saved;
    const Tensor upstream = is_narrow_float(g.dtype()) || is_low_precision_float(g.dtype()) ? cast(g, DType::Float32) : g;
    accumulate(x, mul(values, sub(upstream, sum(mul(upstream, values), dim, true))));
  });
}
Tensor attention_softmax(const Tensor& scores, const Tensor& mask, double divisor, int dim) {
  if (autocast::should_cast({&scores, &mask}, DType::Float32))
    return attention_softmax(cast(scores, DType::Float32), cast(mask, DType::Float32), divisor, dim);
  if (is_low_precision_float(scores.dtype()) || is_low_precision_float(mask.dtype())) {
    require(scores.dtype() == mask.dtype(), "Low precision attention operands must match");
    require(broadcast(scores.shape(), mask.shape()) == scores.shape(), "Attention mask may not expand scores");
    if (scores.shape().empty()) { require(dim == -1 || dim == 0, "Invalid scalar attention axis"); dim = 0; }
    else dim = axis(dim, scores.shape().size());
    divisor = float(divisor);
    return tracked(dispatch_tensor("attention_softmax", scores.shape(), {scores, mask}, {dim, float_bits(float(divisor))}, scores.dtype()), {scores, mask}, [scores, mask, divisor, dim](const Tensor& g) {
      Tensor gradient = low_attention_softmax_derivative(scores, mask, divisor, dim, g);
      if (scores.requires_grad()) accumulate(scores, div(gradient, divisor));
      if (mask.requires_grad()) accumulate(mask, low_attention_reduce_shape(gradient, mask.shape()));
    });
  }
  require(is_arithmetic_float(scores.dtype()) && mask.dtype() == scores.dtype(),
          "Attention softmax requires matching floating input and mask dtypes");
  require(broadcast(scores.shape(), mask.shape()) == scores.shape(),
          "Attention mask must broadcast to the score shape");
  if (scores.shape().empty()) {
    require(dim == 0 || dim == -1, "Scalar attention softmax axis out of range");
    dim = 0;
  } else dim = axis(dim, scores.shape().size());
  std::vector<int> attributes{dim, float_bits(float(divisor))};
  if (scores.dtype() == DType::Float64) {
    attributes.resize(8);
    std::memcpy(attributes.data() + 5, &divisor, sizeof(divisor));
    attributes[7] = 1;
  } else divisor = float(divisor);
  Tensor result = dispatch_tensor("attention_softmax", scores.shape(), {scores, mask}, attributes, scores.dtype());
  return tracked(result, {scores, mask}, [scores, mask, divisor, dim](const Tensor& g) {
    // Recompute probabilities in the operation's accumulation dtype.
    const DType dtype = scores.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor probability = softmax(cast(add(div(scores, divisor), mask), dtype), dim);
    Tensor upstream = cast(g, dtype);
    Tensor gradient = cast(mul(probability, sub(upstream, sum(mul(upstream, probability), dim, true))), scores.dtype());
    if (scores.requires_grad()) accumulate(scores, div(gradient, divisor));
    if (mask.requires_grad()) accumulate(mask, unbroadcast(gradient, mask.shape()));
  });
}
Tensor tiled_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                       const Tensor& mask, double divisor, int query_tile, bool causal, int causal_offset, AttentionTiling tiling) {
  if (is_low_precision_float(query.dtype()) || is_low_precision_float(key.dtype()) || is_low_precision_float(value.dtype()) || is_low_precision_float(mask.dtype())) {
    require(query.dtype() == key.dtype() && query.dtype() == value.dtype() && query.dtype() == mask.dtype(), "Low precision attention operands must match");
    require(query.shape().size() == 4 && value.shape().size() == 4, "Tiled attention requires rank-four inputs");
    divisor = float(divisor);
    Tensor result = dispatch_tensor("tiled_attention", {query.shape()[0], query.shape()[1], query.shape()[2], value.shape()[3]}, {query, key, value, mask},
                                  {float_bits(float(divisor)), query_tile, int(causal), causal_offset, int(tiling)}, query.dtype());
    return tracked(result, {query, key, value, mask}, [query, key, value, mask, divisor, causal, causal_offset](const Tensor& g) {
      std::vector<Tensor> gradients = low_tiled_attention_derivatives(query, key, value, mask, divisor, causal, causal_offset, g);
      const Tensor parents[]{query, key, value, mask};
      for (int i = 0; i < 4; ++i) if (gradients[i].defined()) accumulate(parents[i], gradients[i]);
    });
  }
  require(query.shape().size() == 4 && key.shape().size() == 4 && value.shape().size() == 4,
          "Tiled attention requires rank-four query, key and value");
  require(is_arithmetic_float(query.dtype()) && key.dtype() == query.dtype() && value.dtype() == query.dtype() && mask.dtype() == query.dtype(),
          "Tiled attention requires matching floating dtypes");
  Shape q = query.shape(), k = key.shape(), v = value.shape();
  require(q[0] == k[0] && q[0] == v[0] && q[1] > 0 && k[1] > 0 && q[1] % k[1] == 0 && k[1] == v[1] &&
          q[3] == k[3] && k[2] == v[2], "Incompatible tiled attention dimensions");
  require(query_tile > 0, "Tiled attention query tile must be positive");
  require(tiling == AttentionTiling::Queries || tiling == AttentionTiling::Heads, "Invalid attention tiling mode");
  Shape scores{q[0], q[1], q[2], k[2]};
  require(broadcast(scores, mask.shape()) == scores, "Tiled attention mask must broadcast to the scores");
  std::vector<int> attributes{float_bits(float(divisor)), query_tile, int(causal), causal_offset, int(tiling)};
  if (query.dtype() == DType::Float64) {
    attributes.resize(8);
    std::memcpy(attributes.data() + 5, &divisor, sizeof(divisor));
    attributes[7] = 1;
  } else divisor = float(divisor);
  Tensor result = dispatch_tensor("tiled_attention", {q[0], q[1], q[2], v[3]}, {query, key, value, mask}, attributes, query.dtype());
  return tracked(result, {query, key, value, mask}, [query, key, value, mask, divisor, causal, causal_offset](const Tensor& g) {
    const Shape& q = query.shape();
    const Shape& k = key.shape();
    int groups = q[1] / k[1];
    LocalFunction expand_heads{[&](const Tensor& x) {
      const Shape& s = x.shape();
      return reshape(expand(reshape(x, {s[0], s[1], 1, s[2], s[3]}), {s[0], s[1], groups, s[2], s[3]}),
                     {s[0], q[1], s[2], s[3]});
    }};
    LocalFunction reduce_heads{[&](const Tensor& gradient, const Shape& s) {
      return reshape(sum(reshape(gradient, {s[0], s[1], groups, s[2], s[3]}), 2), s);
    }};
    Tensor keys = expand_heads(key);
    Tensor upper, effective_mask = mask;
    if (causal && q[2] && k[2] && causal_offset < k[2] - 1) {
      upper = causal_offset < 1 - q[2] ? Tensor::ones({}, query.backend(), false, DType::Bool) :
          triu(Tensor::ones({q[2], k[2]}, query.backend(), false, DType::Bool), causal_offset + 1);
      float minimum = query.dtype() == DType::Float16 ? -65504.0f :
                      query.dtype() == DType::BFloat16 ? -3.3895313892515355e38f : -3.402823466e38f;
      Tensor excluded = query.dtype() == DType::Float64 ?
          Tensor::from_double_data({}, {-std::numeric_limits<double>::max()}, query.backend()) :
          Tensor::from_data({}, {minimum}, query.backend(), false, query.dtype());
      effective_mask = where(upper, excluded, mask);
    }
    const DType dtype = query.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor probability = softmax(cast(add(div(matmul_transposed(query, keys), divisor), effective_mask), dtype), -1);
    Tensor rounded = cast(probability, query.dtype());
    if (value.requires_grad()) accumulate(value, reduce_heads(matmul(transpose(rounded, 2, 3), g), value.shape()));
    if (query.requires_grad() || key.requires_grad() || mask.requires_grad()) {
      Tensor upstream = cast(matmul_transposed(g, expand_heads(value)), dtype);
      Tensor masked_gradient = cast(mul(probability, sub(upstream, sum(mul(upstream, probability), -1, true))), query.dtype());
      if (mask.requires_grad()) {
        Tensor mask_gradient = upper.defined() ? where(upper, Tensor::zeros({}, query.backend(), false, query.dtype()), masked_gradient) : masked_gradient;
        accumulate(mask, unbroadcast(mask_gradient, mask.shape()));
      }
      Tensor scaled_gradient = div(masked_gradient, divisor);
      if (query.requires_grad()) accumulate(query, matmul(scaled_gradient, keys));
      if (key.requires_grad()) accumulate(key, reduce_heads(matmul(transpose(scaled_gradient, 2, 3), query), key.shape()));
    }
  });
}

Tensor log_softmax(const Tensor& x, int dim) {
  require(is_differentiable_float(x.dtype()), "Log softmax requires floating input");
  if (x.shape().empty()) {
    require(dim == 0 || dim == -1, "Scalar log_softmax axis out of range");
    return reshape(log_softmax(reshape(x, {1}), 0), {});
  }
  dim = axis(dim, x.shape().size());
  if (is_low_precision_float(x.dtype())) {
    Tensor saved = dispatch_tensor("log_softmax", x.shape(), {x}, {dim, 1}, DType::Float32);
    Tensor y = cast(saved, x.dtype());
    return tracked(y, {x}, [x, saved, dim](const Tensor& g) {
      accumulate(x, dispatch_tensor("log_softmax", x.shape(), {saved, g}, {dim, 2}, x.dtype()));
    });
  }
  Tensor y = dispatch_tensor("log_softmax", x.shape(), {x}, {dim}, x.dtype()),
         saved = y.detach();
  return tracked(y, {x}, [x, saved, dim](const Tensor& g) {
    const Tensor values = is_narrow_float(saved.dtype()) ? cast(saved, DType::Float32) : saved;
    const Tensor upstream = is_narrow_float(g.dtype()) || is_low_precision_float(g.dtype()) ? cast(g, DType::Float32) : g;
    accumulate(x, sub(upstream, mul(exp(values), sum(upstream, dim, true))));
  });
}
static Tensor cuda_batch_norm_reduce(const Tensor& values, DType dtype, int batch, int spatial, bool input_gradient) {
  // Round each product and each shuffled BatchNorm
  // partial to the input dtype, but accumulates each lane in Float32.
  int columns = 32;
  while (columns < spatial && columns < 512) columns *= 2;
  int rows = 512 / columns, channels = values.shape()[0];
  if (!input_gradient) {
    rows = 1;
    while (rows < 16 && rows * 2 <= batch) rows *= 2;
    columns = std::min(columns, 512 / rows);
  }
  int threads = rows * columns, warps = threads / 32;
  int64_t padded_batch = (int64_t(batch) + rows - 1) / rows * rows;
  int64_t padded_spatial = (int64_t(spatial) + columns - 1) / columns * columns;
  require(padded_batch <= std::numeric_limits<int>::max() && padded_spatial <= std::numeric_limits<int>::max(),
          "Batch normalization reduction padding exceeds Int32 dimensions");
  Tensor padded = reshape(cast(cast(values, dtype), DType::Float32), {channels, batch, spatial});
  if (padded_spatial != spatial)
    padded = concat({padded, Tensor::zeros({channels, batch, int(padded_spatial) - spatial}, values.backend())}, 2);
  if (padded_batch != batch)
    padded = concat({padded, Tensor::zeros({channels, int(padded_batch) - batch, int(padded_spatial)}, values.backend())}, 1);
  Tensor lanes = reshape(permute(reshape(padded, {channels, int(padded_batch) / rows, rows, int(padded_spatial) / columns, columns}),
                               {0, 2, 4, 1, 3}), {channels, threads, -1});
  LocalFunction warp{[dtype](Tensor partials) {
    for (int offset = 16; offset; offset /= 2)
      partials = add(slice(partials, -1, 0, offset), cast(cast(slice(partials, -1, offset, offset * 2), dtype), DType::Float32));
    return partials;
  }};
  Tensor partials = reshape(warp(reshape(sum(lanes, 2), {channels, warps, 32})), {channels, warps});
  return warp(concat({partials, Tensor::zeros({channels, 32 - warps}, values.backend())}, 1));
}
static Tensor narrow_affine_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                                 float eps, int kind, int groups) {
  // Accumulate in Float32 and store statistics at the precision selected for each backend.
  // CPU Layer/BatchNorm and both GroupNorm backends round same-dtype statistics.
  const Tensor input = cast(x.detach(), DType::Float32);
  const Tensor weight = cast(w.detach(), DType::Float32);
  const Tensor bias = cast(b.detach(), DType::Float32);
  if (kind == 2 && x.backend() == Backend::CUDA) {
    const Tensor epsilon = Tensor::from_data({}, {eps}, Backend::CPU, false, x.dtype());
    eps = epsilon.data()[0];
  }
  std::vector<int> axes(x.shape().size());
  std::iota(axes.begin(), axes.end(), 0);
  if (kind == 1) std::swap(axes[0], axes[1]);
  Shape affine(x.shape().size(), 1);
  if (kind != 0) affine[1] = x.shape()[1];
  Shape grouped = kind == 0 ? Shape{-1, int(w.numel())} :
                  kind == 1 ? Shape{x.shape()[1], -1} : Shape{x.shape()[0], groups, -1};
  const Tensor values = reshape(kind == 1 ? permute(input, axes) : input, grouped);
  const Tensor weights = kind == 0 ? reshape(weight, {1, int(w.numel())}) :
                         kind == 1 ? unsqueeze(weight, 1) : reshape(expand(reshape(weight, affine), x.shape()), values.shape());
  const Tensor biases = kind == 0 ? reshape(bias, {1, int(b.numel())}) :
                        kind == 1 ? unsqueeze(bias, 1) : reshape(expand(reshape(bias, affine), x.shape()), values.shape());
  const Tensor average = mean(values, -1, true);
  const Tensor centered = sub(values, average);
  const Tensor inverse = rsqrt(add(mean(mul(centered, centered), -1, true), eps));
  LocalFunction restore{[x, axes, kind](const Tensor& value) {
    if (kind != 1) return reshape(value, x.shape());
    Shape shape = x.shape();
    std::swap(shape[0], shape[1]);
    return permute(reshape(value, shape), axes);
  }};
  Tensor output = kind == 2 && x.backend() == Backend::CUDA ?
      dispatch_tensor("group_norm", x.shape(), {x, w, b}, {groups, float_bits(eps)}, x.dtype()) :
      cast(restore(add(mul(mul(centered, inverse), weights), biases)), x.dtype());
  const DType statistic_dtype = kind == 2 || x.backend() == Backend::CPU ? w.dtype() : DType::Float32;
  const Tensor saved_mean = cast(cast(average, statistic_dtype), DType::Float32);
  const Tensor saved_inverse = cast(cast(inverse, statistic_dtype), DType::Float32);
  return tracked(output, {x, w, b}, [x, w, b, values, weights, saved_mean, saved_inverse, axes, kind, restore, eps, groups](const Tensor& g) {
    Tensor upstream = cast(g, DType::Float32);
    upstream = reshape(kind == 1 ? permute(upstream, axes) : upstream, values.shape());
    if (kind == 1 && x.backend() == Backend::CUDA) {
      int batch = x.shape()[0], channels = x.shape()[1], spatial = int(x.numel() / batch / channels);
      bool channels_last = spatial == 1 || (channels == 1 && (x.shape().size() == 4 || x.shape().size() == 5));
      // The cuDNN-compatible path handles ordinary FP16 inputs with Float32
      // affine parameters. Its native contiguous path has narrow shuffle sums.
      bool native = x.dtype() == DType::BFloat16 || w.dtype() == x.dtype() || batch > 880801 ||
                    x.numel() == size_t(std::numeric_limits<int>::max());
      if (native && !channels_last) {
        Tensor reduced = cuda_batch_norm_reduce(concat({upstream, mul(upstream, sub(values, saved_mean))}, 0),
                                              x.dtype(), batch, spatial, x.requires_grad());
        Tensor total = slice(reduced, 0, 0, channels), dot = slice(reduced, 0, channels, channels * 2);
        if (x.requires_grad()) {
          float norm = 1.0f / float(values.shape()[1]);
          Tensor projection = mul(mul(mul(dot, norm), saved_inverse), saved_inverse);
          Tensor gradient = mul(sub(sub(upstream, mul(sub(values, saved_mean), projection)), mul(total, norm)),
                              mul(saved_inverse, weights));
          accumulate(x, restore(gradient));
        }
        if (w.requires_grad()) accumulate(w, reshape(mul(dot, saved_inverse), w.shape()));
        if (b.requires_grad()) accumulate(b, reshape(total, b.shape()));
        return;
      }
    }
    const Tensor weighted = mul(upstream, weights);
    const Tensor db = mean(weighted, -1, true);
    const Tensor ds = mean(mul(weighted, values), -1, true);
    const Tensor c2 = mul(sub(mul(db, saved_mean), ds), pow(saved_inverse, 3));
    const Tensor c3 = sub(mul(mul(c2, saved_mean), -1), mul(db, saved_inverse));
    if (x.requires_grad()) {
      if (kind == 2) accumulate(x, dispatch_tensor("group_norm_input_grad", x.shape(), {x, w, g},
                                                  {groups, float_bits(eps)}, x.dtype()));
      else accumulate(x, restore(add(add(mul(weighted, saved_inverse), mul(c2, values)), c3)));
    }
    LocalFunction affine_reduce{[kind, x, w, restore](const Tensor& value) {
      if (kind == 0) {
        if (x.backend() == Backend::CPU) {
          // CPU LayerNorm stores each row update in an input-dtype partial
          // buffer, including when the final affine gradients are Float32.
          Tensor partial = Tensor::zeros({1, value.shape()[1]}, x.backend(), false, x.dtype());
          for (int row = 0; row < value.shape()[0]; ++row)
            partial = cast(add(cast(partial, DType::Float32), slice(value, 0, row, row + 1)), x.dtype());
          return reshape(partial, w.shape());
        }
        return reshape(sum(value, 0), w.shape());
      }
      const Tensor full = restore(value);
      std::vector<int> order(x.shape().size());
      std::iota(order.begin(), order.end(), 0);
      std::swap(order[0], order[1]);
      return sum(reshape(permute(full, order), {x.shape()[1], -1}), 1);
    }};
    if (w.requires_grad()) accumulate(w, affine_reduce(mul(mul(upstream, sub(values, saved_mean)), saved_inverse)));
    if (b.requires_grad()) accumulate(b, affine_reduce(upstream));
  });
}
static Tensor float64_affine_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                                  double eps, int kind, int groups) {
  // Keep statistics and gradients in Double. The composition dispatches every
  // arithmetic operation to the input's native CPU or CUDA backend.
  std::vector<int> axes(x.shape().size());
  std::iota(axes.begin(), axes.end(), 0);
  if (kind == 1) std::swap(axes[0], axes[1]);
  size_t rows = 0, width = 0;
  if (kind == 0) {
    rows = shape_element_count(Shape(x.shape().begin(), x.shape().end() - w.shape().size()));
    width = w.numel();
  } else if (kind == 1) {
    rows = x.shape()[1];
    width = x.numel() / rows;
  } else {
    rows = shape_element_count({x.shape()[0], groups});
    width = shape_element_count(Shape(x.shape().begin() + 1, x.shape().end())) / groups;
  }
  require(rows <= size_t(INT32_MAX) && width <= size_t(INT32_MAX), "Double normalization grouping exceeds Int32");
  const Tensor values = reshape(kind == 1 ? permute(x.detach(), axes) : x.detach(), {int(rows), int(width)});
  Shape affine(x.shape().size(), 1);
  if (kind != 0) affine[1] = x.shape()[1];
  LocalFunction grouped_affine{[&](const Tensor& parameter) {
    if (kind == 0) return reshape(parameter.detach(), {1, int(width)});
    if (kind == 1) return reshape(parameter.detach(), {int(rows), 1});
    return reshape(expand(reshape(parameter.detach(), affine), x.shape()), values.shape());
  }};
  const Tensor weights = grouped_affine(w), biases = grouped_affine(b);
  const Tensor average = mean(values, -1, true), centered = sub(values, average);
  const Tensor variance = mean(mul(centered, centered), -1, true);
  Tensor inverse = rsqrt(add(variance, eps));
  if (kind == 1 && x.backend() == Backend::CPU && eps == 0) {
    // Native CPU BatchNorm defines zero inverse deviation for a constant row
    // with zero epsilon; the CUDA reference instead propagates NaN.
    Tensor zero = Tensor::zeros({}, x.backend(), false, DType::Float64);
    inverse = where(eq(variance, zero), zero, inverse);
  }
  const Tensor normalized = mul(centered, inverse);
  LocalFunction restore{[x, axes, kind](const Tensor& value) {
    if (kind != 1) return reshape(value, x.shape());
    Shape shape = x.shape();
    std::swap(shape[0], shape[1]);
    return permute(reshape(value, shape), axes);
  }};
  const size_t spatial = kind == 2 ? shape_element_count(Shape(x.shape().begin() + 2, x.shape().end())) : 1;
  LocalFunction multiply_add{[x](const Tensor& a, const Tensor& b, const Tensor& c) {
    // Preserve the fused CUDA affine arithmetic and the CPU rounding boundaries.
    if (x.backend() == Backend::CUDA)
      return dispatch_tensor("float64_fma", broadcast(broadcast(a.shape(), b.shape()), c.shape()), {a, b, c}, {}, DType::Float64);
    return add(mul(a, b), c);
  }};
  Tensor affine_output;
  if ((kind == 1 && x.backend() == Backend::CPU) ||
      (kind == 2 && (x.backend() == Backend::CPU || spatial != 1))) {
    Tensor scale = mul(inverse, weights);
    Tensor offset = multiply_add(mul(scale, -1.), average, biases);
    affine_output = multiply_add(scale, values, offset);
  } else affine_output = add(mul(normalized, weights), biases);
  Tensor output = restore(affine_output);
  return tracked(output, {x, w, b}, [x, w, b, kind, groups, spatial, values, weights, average, inverse, normalized, axes, restore, multiply_add](const Tensor& g) {
    Tensor upstream = reshape(kind == 1 ? permute(g, axes) : g, values.shape());
    if (kind == 2) {
      const int rows = values.shape()[0], channels = x.shape()[1], per_group = channels / groups;
      Tensor ds = sum(reshape(mul(upstream, values), {rows, per_group, int(spatial)}), -1);
      Tensor db = sum(reshape(upstream, {rows, per_group, int(spatial)}), -1);
      Tensor gamma = reshape(expand(reshape(w.detach(), {1, groups, per_group}),
                                  {x.shape()[0], groups, per_group}), {rows, per_group});
      if (x.requires_grad()) {
        Tensor ds_weighted = sum(mul(ds, gamma), -1, true);
        Tensor db_weighted = sum(mul(db, gamma), -1, true);
        const double reciprocal_width = 1. / double(values.shape()[1]);
        Tensor c2 = mul(mul(mul(mul(multiply_add(db_weighted, average, mul(ds_weighted, -1.)), inverse), inverse), inverse), reciprocal_width);
        Tensor c3 = multiply_add(mul(c2, -1.), average, mul(mul(mul(db_weighted, inverse), reciprocal_width), -1.));
        Tensor c1 = mul(inverse, weights);
        Tensor gradient = add(multiply_add(c1, upstream, mul(c2, values)), c3);
        accumulate(x, restore(gradient));
      }
      if (w.requires_grad()) {
        Tensor centered_gradient = x.backend() == Backend::CUDA && spatial == 1 ?
            multiply_add(upstream, values, mul(mul(upstream, average), -1.)) :
            multiply_add(mul(db, -1.), average, ds);
        Tensor gradient = mul(centered_gradient, inverse);
        accumulate(w, sum(reshape(gradient, {x.shape()[0], channels}), 0));
      }
      if (b.requires_grad()) accumulate(b, sum(reshape(db, {x.shape()[0], channels}), 0));
      return;
    }
    if (x.requires_grad()) {
      Tensor weighted = mul(upstream, weights);
      Tensor gradient = mul(inverse, sub(sub(weighted, mean(weighted, -1, true)),
                                      mul(normalized, mean(mul(weighted, normalized), -1, true))));
      accumulate(x, restore(gradient));
    }
    LocalFunction affine_reduce{[w, kind](const Tensor& value) {
      if (kind == 0) return reshape(sum(value, 0), w.shape());
      return reshape(sum(value, 1), w.shape());
    }};
    if (w.requires_grad()) accumulate(w, affine_reduce(mul(upstream, normalized)));
    if (b.requires_grad()) accumulate(b, affine_reduce(upstream));
  });
}

static Tensor low_affine_normalization(const std::string& operation, const Tensor& x, const Tensor& w,
                                       const Tensor& b, double epsilon, int groups = 1, int rounding = 0) {
  require(w.dtype() == x.dtype() && (!b.defined() || b.dtype() == x.dtype()), "Low precision normalization requires matching dtypes");
  std::vector<int> attributes(10);
  if (operation == "layer_norm" && x.backend() == Backend::CPU && x.dtype() == DType::Float32) {
    attributes.resize(11);
    attributes[10] = 1;
  }
  attributes[2] = rounding;
  attributes[9] = operation == "batch_norm" ? 1 : 0;
  uint64_t epsilon_bits;
  std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
  attributes[5] = int(uint32_t(epsilon_bits));
  attributes[6] = int(uint32_t(epsilon_bits >> 32));
  attributes[7] = 1;
  if (operation == "batch_norm") attributes[0] = float_bits(float(epsilon));
  else {
    attributes[0] = operation == "group_norm" ? groups : int(w.shape().size());
    attributes[1] = float_bits(float(epsilon));
  }
  std::vector<Tensor> parents{x, w};
  if (b.defined()) parents.push_back(b);
  Tensor output = dispatch_tensor(operation, x.shape(), parents, attributes, x.dtype());
  return tracked(output, parents, [operation, x, w, b, attributes](const Tensor& g) {
    std::vector<int> derivative = attributes;
    if (x.requires_grad()) {
      derivative[8] = 1;
      accumulate(x, dispatch_tensor(operation, x.shape(), {x, w, g}, derivative, x.dtype()));
    }
    derivative[8] = 2;
    if (w.requires_grad()) accumulate(w, dispatch_tensor(operation, w.shape(), {x, g}, derivative, w.dtype()));
    derivative[8] = 3;
    if (b.defined() && b.requires_grad()) accumulate(b, dispatch_tensor(operation, b.shape(), {x, g}, derivative, b.dtype()));
  });
}
Tensor layer_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                  double eps) {
  if (autocast::should_cast({&x, &w, &b}, DType::Float32))
    return layer_norm(cast(x, DType::Float32), cast(w, DType::Float32), cast(b, DType::Float32), eps);
  if (x.dtype() != DType::Float64) eps = float(eps);
  require(eps >= 0, "Normalization epsilon must be nonnegative");
  require(is_differentiable_float(x.dtype()) && (w.dtype() == x.dtype() || (is_narrow_float(x.dtype()) && x.backend() == Backend::CPU && w.dtype() == DType::Float32)) && b.dtype() == w.dtype(), "Layer normalization requires floating input and affine parameters");
  require(!w.shape().empty() && w.shape().size() <= x.shape().size() &&
              Shape(x.shape().end() - w.shape().size(), x.shape().end()) == w.shape() && b.shape() == w.shape(),
          "layer_norm affine shape mismatch");
  if (is_low_precision_float(x.dtype()) || (x.backend() == Backend::CPU && x.dtype() == DType::Float32))
    return low_affine_normalization("layer_norm", x, w, b, eps);
  if (x.dtype() == DType::Float64) return float64_affine_norm(x, w, b, eps, 0, 1);
  if (is_narrow_float(x.dtype())) return narrow_affine_norm(x, w, b, float(eps), 0, 1);
  if (w.shape().size() > 1) {
    size_t rows = 1;
    for (size_t i = 0; i < x.shape().size() - w.shape().size(); ++i) rows *= size_t(x.shape()[i]);
    return reshape(layer_norm(reshape(x, {int(rows), int(w.numel())}), reshape(w, {int(w.numel())}),
                              reshape(b, {int(b.numel())}), eps), x.shape());
  }
  Tensor centered = sub(x, mean(x, -1, true));
  return add(mul(mul(centered,
                     rsqrt(add(mean(mul(centered, centered), -1, true), eps))),
                 w),
             b);
}
Tensor rms_norm(const Tensor& x, const Tensor& w, double eps, RmsNormRounding rounding) {
  if (x.dtype() != DType::Float64) eps = float(eps);
  require(eps >= 0 && !w.shape().empty() && w.shape().size() <= x.shape().size() &&
              Shape(x.shape().end() - w.shape().size(), x.shape().end()) == w.shape(),
          "Invalid RMS normalization parameters");
  require(is_differentiable_float(x.dtype()) && is_differentiable_float(w.dtype()) &&
              (rounding == RmsNormRounding::Output || w.dtype() == x.dtype()),
          "RMS normalization requires floating inputs and matching dtypes when rounding the normalized activation");
  require(rounding == RmsNormRounding::Output || rounding == RmsNormRounding::Normalized,
          "Invalid RMS normalization rounding");
  if (is_low_precision_float(x.dtype()) || is_low_precision_float(w.dtype())) {
    return low_affine_normalization("rms_norm", x, w, Tensor{}, eps, 1, int(rounding));
  }
  if (x.dtype() == DType::Float64 || w.dtype() == DType::Float64) {
    const DType dtype = x.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    size_t rows = shape_element_count(Shape(x.shape().begin(), x.shape().end() - w.shape().size()));
    require(rows <= size_t(INT32_MAX) && w.numel() <= size_t(INT32_MAX), "Double RMS normalization grouping exceeds Int32");
    Tensor values = reshape(cast(x, dtype), {int(rows), int(w.numel())});
    Tensor normalized = mul(values, rsqrt(add(mean(mul(values, values), -1, true), eps)));
    if (rounding == RmsNormRounding::Normalized) normalized = cast(normalized, x.dtype());
    return reshape(cast(mul(normalized, reshape(w, {int(w.numel())})), x.dtype()), x.shape());
  }
  Tensor result = dispatch_tensor("rms_norm", x.shape(), {x, w},
                                {int(w.shape().size()), float_bits(float(eps)), int(rounding)}, x.dtype());
  return tracked(result, {x, w}, [x, w, eps, rounding](const Tensor& g) {
    if (!w.numel() || !x.numel()) {
      if (x.requires_grad()) accumulate(x, Tensor::zeros(x.shape(), x.backend(), false, x.dtype()));
      if (w.requires_grad()) accumulate(w, Tensor::zeros(w.shape(), w.backend(), false, w.dtype()));
      return;
    }
    int width = int(w.numel()), rows = int(x.numel() / w.numel());
    Tensor values = reshape(cast(x, DType::Float32), {rows, width});
    Tensor inverse = rsqrt(add(mean(mul(values, values), -1, true), eps));
    Tensor normalized = mul(values, inverse);
    Tensor upstream = reshape(g, {rows, width});
    Tensor weights = reshape(w, {width});
    bool rounded = rounding == RmsNormRounding::Normalized;
    if (w.requires_grad()) {
      Tensor product = rounded ? mul(upstream, cast(normalized, x.dtype())) :
                               mul(cast(upstream, DType::Float32), normalized);
      accumulate(w, reshape(sum(product, 0), w.shape()));
    }
    if (x.requires_grad()) {
      Tensor weighted = rounded ? cast(mul(upstream, weights), DType::Float32) :
                                mul(cast(upstream, DType::Float32), cast(weights, DType::Float32));
      Tensor inverse_grad = sum(mul(weighted, values), -1, true);
      Tensor variance_grad = mul(mul(inverse_grad, -0.5f), pow(inverse, 3));
      // Keep the mean and square backward rounding boundaries of the eager
      // reference, including the backend-specific mean backward scaling.
      Tensor square_grad = dispatch_tensor("reduce_grad", values.shape(), {values, inverse, variance_grad}, {1, 1, 1});
      Tensor gradient = add(mul(weighted, inverse), mul(square_grad, mul(values, 2.0f)));
      accumulate(x, reshape(gradient, x.shape()));
    }
  });
}
Tensor batch_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                  double eps) {
  if (autocast::should_cast({&x, &w, &b}, DType::Float32))
    return batch_norm(cast(x, DType::Float32), cast(w, DType::Float32), cast(b, DType::Float32), eps);
  if (x.dtype() != DType::Float64 && !is_low_precision_float(x.dtype())) eps = float(eps);
  require(is_differentiable_float(x.dtype()) && (w.dtype() == x.dtype() || (is_narrow_float(x.dtype()) && w.dtype() == DType::Float32)) && b.dtype() == w.dtype(), "Batch normalization requires floating inputs");
  require(x.shape().size() >= 2 && w.shape() == Shape{x.shape()[1]} &&
              b.shape() == w.shape() && ((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? eps >= 0 : eps > 0),
          "Invalid batch normalization parameters");
  require(x.shape()[1] > 0 && ((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.numel() / x.shape()[1] != 1 : x.numel() / x.shape()[1] > 1),
          "Training batch normalization requires multiple values per channel");
  if (is_low_precision_float(x.dtype())) return low_affine_normalization("batch_norm", x, w, b, eps);
  if (x.dtype() == DType::Float64) return float64_affine_norm(x, w, b, eps, 1, 1);
  if (is_narrow_float(x.dtype())) return narrow_affine_norm(x, w, b, float(eps), 1, 1);
  std::vector<int> axes(x.shape().size());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[0], axes[1]);
  Tensor c = reshape(permute(x, axes), {x.shape()[1], -1});
  Tensor centered = sub(c, mean(c, -1, true));
  Tensor y =
      add(mul(mul(centered,
                  rsqrt(add(mean(mul(centered, centered), -1, true), eps))),
              unsqueeze(w, 1)),
          unsqueeze(b, 1));
  Shape s = x.shape();
  std::swap(s[0], s[1]);
  return permute(reshape(y, s), axes);
}
Tensor group_norm(const Tensor& x, int groups, const Tensor& w, const Tensor& b,
                  double eps) {
  if (autocast::should_cast({&x, &w, &b}, DType::Float32))
    return group_norm(cast(x, DType::Float32), groups, cast(w, DType::Float32), cast(b, DType::Float32), eps);
  require(is_differentiable_float(x.dtype()) && (w.dtype() == x.dtype() || (is_narrow_float(x.dtype()) && x.backend() == Backend::CPU && w.dtype() == DType::Float32)) && b.dtype() == w.dtype(), "Group normalization requires floating inputs");
  require(x.shape().size() >= 2 && groups > 0 && x.shape()[1] % groups == 0 &&
              w.shape() == Shape{x.shape()[1]} && b.shape() == w.shape() &&
              ((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? eps >= 0 : eps > 0),
          "Invalid group normalization parameters");
  require((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.numel() / groups != 1 : x.numel() / groups > 1,
          "Group normalization requires multiple values per channel group across the batch");
  if (is_low_precision_float(x.dtype())) {
    require(x.backend() == Backend::CPU || x.shape()[1] > 0, "CUDA group normalization requires nonzero channels");
    return low_affine_normalization("group_norm", x, w, b, eps, groups);
  }
  if (x.dtype() == DType::Float64) {
    require(x.backend() == Backend::CPU || x.shape()[1] > 0, "CUDA group normalization requires nonzero channels");
    return float64_affine_norm(x, w, b, eps, 2, groups);
  }
  if (is_narrow_float(x.dtype())) return narrow_affine_norm(x, w, b, float(eps), 2, groups);
  Shape affine(x.shape().size(), 1);
  affine[1] = x.shape()[1];
  std::vector<int> attributes = group_norm_attributes(groups, eps);
  Tensor output = dispatch_tensor("group_norm", x.shape(), {x, w, b}, attributes, x.dtype());
  return tracked(output, {x, w, b}, [x, w, b, affine, attributes](const Tensor& gradient) {
    if (x.requires_grad()) accumulate(x, dispatch_tensor("group_norm_input_grad", x.shape(), {x, w, gradient}, attributes, x.dtype()));
    if (w.requires_grad()) accumulate(w, dispatch_tensor("group_norm_weight_grad", w.shape(), {x, gradient}, attributes, w.dtype()));
    if (b.requires_grad()) accumulate(b, reshape(unbroadcast(gradient, affine), b.shape()));
  });
}
Tensor mse_loss(const Tensor& a, const Tensor& b) {
  if (autocast::should_cast({&a, &b}, DType::Float32)) return mse_loss(cast(a, DType::Float32), cast(b, DType::Float32));
  if (is_low_precision_float(a.dtype()) || is_low_precision_float(b.dtype())) {
    require(a.dtype() == b.dtype() && a.backend() == b.backend(), "Low precision MSE requires matching dtypes and backends");
    const Shape shape = broadcast(a.shape(), b.shape());
    const Tensor left = a.shape() == shape ? a.detach() : expand(a.detach(), shape);
    const Tensor right = b.shape() == shape ? b.detach() : expand(b.detach(), shape);
    return tracked(dispatch_tensor("mse_loss", {}, {left, right}, {}, a.dtype()), {a, b}, [a, b, left, right](const Tensor& gradient) {
      if (a.requires_grad()) accumulate(a, unbroadcast(dispatch_tensor("mse_loss", left.shape(), {left, right, gradient}, {1}, DType::Float32), a.shape()));
      if (b.requires_grad()) accumulate(b, unbroadcast(dispatch_tensor("mse_loss", right.shape(), {right, left, gradient}, {1}, DType::Float32), b.shape()));
    });
  }
  if (is_narrow_float(a.dtype()) && a.dtype() == b.dtype())
    return cast(mse_loss(cast(a, DType::Float32), cast(b, DType::Float32)), a.dtype());
  require(a.shape() == b.shape(), "MSE shape mismatch");
  Tensor d = sub(a, b);
  return mean_all(mul(d, d));
}
Tensor soft_target_cross_entropy(const Tensor& logits, const Tensor& probabilities) {
  require(logits.shape() == probabilities.shape() && logits.shape().size() >= 2 && logits.numel() > 0,
          "Soft cross entropy expects matching nonempty logits and probabilities");
  require(logits.dtype() == probabilities.dtype() && is_differentiable_float(logits.dtype()) &&
          logits.backend() == probabilities.backend(), "Soft cross entropy requires matching floating tensors");
  const DType dtype = logits.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
  Tensor log_probs = log_softmax(cast(logits, dtype), -1);
  return mul(mean_all(sum(mul(cast(probabilities, dtype), log_probs), -1)), -1);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
  if (autocast::should_cast({&logits}, DType::Float32)) return cross_entropy(cast(logits, DType::Float32), targets);
  require(logits.backend() == targets.backend(), "Cross entropy requires matching backends");
  require(logits.shape().size() >= 2,
          "Cross entropy requires logits with a class dimension");
  int classes = logits.shape().back();
  require(classes > 0, "Cross entropy requires at least one class");
  require(targets.numel() == logits.numel() / classes,
          "Cross entropy target size mismatch");
  validate_indices(targets, classes);
  if (logits.numel() == 0 && !is_low_precision_float(logits.dtype())) return mean_all(logits);
  Tensor x = reshape(logits, {-1, classes}), y = reshape(targets, {-1});
  Tensor loss = dispatch_tensor("cross_entropy", {}, {x, y}, {}, x.dtype());
  return tracked(loss, {x}, [x, y](const Tensor& g) {
    accumulate(x, dispatch_tensor("cross_entropy_grad", x.shape(), {x, y, g}, {}, x.dtype()));
  });
}
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride,
              int padding) {
  if (const DType target = autocast::dtype(); autocast::should_cast({&x, &w, &b}, target))
    return conv2d(cast(x, target), cast(w, target), autocast::cast_optional(b, target), stride, padding);
  if (is_complex(x.dtype()) || is_complex(w.dtype()) || (b.defined() && is_complex(b.dtype()))) {
    require(x.dtype() == w.dtype() && is_complex(x.dtype()) && (!b.defined() || b.dtype() == x.dtype()),
            "Complex convolution requires matching dtypes");
    if (x.shape().size() == 3) return squeeze(conv2d(unsqueeze(x, 0), w, b, stride, padding), 0);
    const Tensor values = x.dtype() == DType::Complex32 ? cast(x, DType::Complex64) : x;
    const Tensor weights = w.dtype() == DType::Complex32 ? cast(w, DType::Complex64) : w;
    const Tensor xr = real(values), xi = imag(values), wr = real(weights), wi = imag(weights);
    const Tensor bias = b.defined() ? cast(b, values.dtype()) : Tensor{};
    const Tensor br = bias.defined() ? real(bias) : Tensor{};
    const Tensor bi = bias.defined() ? imag(bias) : Tensor{};
    const Tensor a = conv2d(xr, wr, br, stride, padding);
    const Tensor c = conv2d(xi, wi, {}, stride, padding);
    const Tensor d = conv2d(add(xr, xi), add(wr, wi), bias.defined() ? add(br, bi) : Tensor{}, stride, padding);
    return cast(complex(sub(a, c), sub(sub(d, a), c)), x.dtype());
  }
  if ((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) && x.shape().size() == 3)
    return squeeze(conv2d(unsqueeze(x, 0), w, b, stride, padding), 0);
  require((is_arithmetic_float(x.dtype()) || is_low_precision_float(x.dtype())) && w.dtype() == x.dtype() &&
              (!b.defined() || b.dtype() == x.dtype()), "Convolution requires matching floating dtypes");
  require(x.shape().size() == 4 && w.shape().size() == 4 &&
              x.shape()[1] == w.shape()[1] && stride > 0 && padding >= 0,
          "Invalid convolution shapes or attributes");
  require(!b.defined() || b.shape() == Shape{w.shape()[0]},
          "Convolution bias shape mismatch");
  const int64_t padded_h = int64_t(x.shape()[2]) + 2 * int64_t(padding),
                padded_w = int64_t(x.shape()[3]) + 2 * int64_t(padding);
  require(padded_h >= w.shape()[2] && padded_w >= w.shape()[3],
          "Convolution kernel exceeds padded input");
  const int64_t h = (padded_h - w.shape()[2]) / stride + 1, ww = (padded_w - w.shape()[3]) / stride + 1;
  require(h <= INT32_MAX && ww <= INT32_MAX, "Convolution output dimensions exceed Int32");
  Shape s{x.shape()[0], (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) && x.shape()[1] == 0 ? 0 : w.shape()[0], int(h), int(ww)};
  std::vector<int> attrs{stride, stride, padding, padding, 1, 1, 1};
  return tracked(dispatch_tensor("conv2d", s, {x, w, b}, attrs, x.dtype()), {x, w, b},
                 [x, w, b, attrs](const Tensor& g) {
                   if (x.requires_grad()) {
                     accumulate(x, dispatch_tensor("conv2d_input_grad",
                                                   x.shape(), {g, w}, attrs, (x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) ? x.dtype() : DType::Float32));
                   }
                   if (w.requires_grad()) {
                     accumulate(w, dispatch_tensor("conv2d_weight_grad",
                                                   w.shape(), {x, g}, attrs, (w.dtype() == DType::Float64 || is_low_precision_float(w.dtype())) ? w.dtype() : DType::Float32));
                   }
                   if (b.requires_grad()) {
                     if ((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) && x.shape()[1] == 0)
                       accumulate(b, Tensor::zeros(b.shape(), b.backend(), false, b.dtype()));
                     else accumulate(b, sum(sum(sum((is_narrow_float(g.dtype()) || is_low_precision_float(g.dtype())) ? cast(g, DType::Float32) : g, 3), 2), 0));
                   }
                 });
}
static Tensor pool(const std::string& op, const Tensor& x, int kernel,
                   int stride, int padding) {
  if ((x.dtype() == DType::Float64 || is_low_precision_float(x.dtype())) && x.shape().size() == 3)
    return squeeze(pool(op, unsqueeze(x, 0), kernel, stride, padding), 0);
  if (stride == 0) {
    stride = kernel;
  }
  require(x.shape().size() == 4 && kernel > 0 && stride > 0 && padding >= 0 &&
              padding <= kernel / 2,
          "Invalid pooling parameters");
  const int64_t padded_h = int64_t(x.shape()[2]) + 2 * int64_t(padding),
                padded_w = int64_t(x.shape()[3]) + 2 * int64_t(padding);
  require(padded_h >= kernel && padded_w >= kernel,
          "Pooling kernel exceeds padded input");
  const int64_t height = (padded_h - kernel) / stride + 1, width = (padded_w - kernel) / stride + 1;
  require(height <= INT32_MAX && width <= INT32_MAX, "Pooling output dimensions exceed Int32");
  Shape s = x.shape();
  s[2] = int(height);
  s[3] = int(width);
  std::vector<int> attrs{kernel, kernel, stride, stride, padding, padding, 1};
  return tracked(
      dispatch_tensor(op, s, {x}, attrs, x.dtype()), {x}, [x, attrs, op](const Tensor& g) {
        accumulate(x, dispatch_tensor(op + "_grad", x.shape(), {x, g}, attrs, x.dtype()));
      });
}
Tensor max_pool2d(const Tensor& x, int k, int s, int p) {
  return pool("max_pool2d", x, k, s, p);
}
Tensor avg_pool2d(const Tensor& x, int k, int s, int p) {
  return pool("avg_pool2d", x, k, s, p);
}
}  // namespace elman
