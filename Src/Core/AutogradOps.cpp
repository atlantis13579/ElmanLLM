#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "Core/Autograd.h"
#include "Core/TensorDispatch.h"

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
    out[out.size() - 1 - k] = std::max(x, y);
  }
  return out;
}
static Tensor unbroadcast(Tensor g, const Shape& shape) {
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
Tensor add(const Tensor& a, float b) {
  return add(a, scalar(b, a.backend()));
}
Tensor sub(const Tensor& a, float b) {
  return sub(a, scalar(b, a.backend()));
}
Tensor mul(const Tensor& a, float b) {
  return mul(a, scalar(b, a.backend()));
}
Tensor div(const Tensor& a, float b) {
  return div(a, scalar(b, a.backend()));
}
Tensor pow(const Tensor& a, float b) {
  return pow(a, scalar(b, a.backend()));
}
static Tensor binary(const std::string& op, const Tensor& a, const Tensor& b) {
  Shape s = broadcast(a.shape(), b.shape());
  Tensor out = dispatch_tensor(op, s, {a, b});
  return tracked(out, {a, b}, [a, b, op](const Tensor& g) {
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
      gb = mul(div(mul(g, a), mul(b, b)), -1);
    } else if (op == "pow") {
      if (a.requires_grad()) {
        ga = mul(g, mul(b, pow(a, sub(b, 1))));
      }
      if (b.requires_grad()) {
        gb = mul(g, mul(pow(a, b), log(a)));
      }
    } else {
      bool mx = op == "maximum";
      Tensor tie = mul(cast(eq(a, b), DType::Float32), 0.5f);
      ga = mul(g, add(cast(mx ? gt(a, b) : lt(a, b), DType::Float32), tie));
      gb = mul(g, add(cast(mx ? lt(a, b) : gt(a, b), DType::Float32), tie));
    }
    if (a.requires_grad()) {
      accumulate(a, unbroadcast(ga, a.shape()));
    }
    if (b.requires_grad()) {
      accumulate(b, unbroadcast(gb, b.shape()));
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

Tensor eq(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("eq", broadcast(a.shape(), b.shape()), {a, b}, {},
                         DType::Bool);
}
Tensor ne(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("ne", broadcast(a.shape(), b.shape()), {a, b}, {},
                         DType::Bool);
}
Tensor gt(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("gt", broadcast(a.shape(), b.shape()), {a, b}, {},
                         DType::Bool);
}
Tensor ge(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("ge", broadcast(a.shape(), b.shape()), {a, b}, {},
                         DType::Bool);
}
Tensor lt(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("lt", broadcast(a.shape(), b.shape()), {a, b}, {},
                         DType::Bool);
}
Tensor le(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("le", broadcast(a.shape(), b.shape()), {a, b}, {},
                         DType::Bool);
}
Tensor logical_and(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("logical_and", broadcast(a.shape(), b.shape()), {a, b},
                         {}, DType::Bool);
}
Tensor logical_or(const Tensor& a, const Tensor& b) {
  return dispatch_tensor("logical_or", broadcast(a.shape(), b.shape()), {a, b},
                         {}, DType::Bool);
}

Tensor logical_not(const Tensor& x) {
  return dispatch_tensor("logical_not", x.shape(), {x}, {}, DType::Bool);
}
Tensor cast(const Tensor& x, DType dtype) {
  Tensor y = dispatch_tensor("cast", x.shape(), {x}, {}, dtype);
  if (x.dtype() == DType::Float32 && dtype == DType::Float32) {
    return tracked(y, {x}, [x](const Tensor& g) { accumulate(x, g); });
  }
  return y;
}
static Tensor unary(const std::string& op, const Tensor& x,
                    float slope = 0.01f) {
  Tensor out = dispatch_tensor(op, x.shape(), {x}, {float_bits(slope)}),
         saved = out.detach();
  return tracked(out, {x}, [x, saved, op, slope](const Tensor& g) {
    Tensor d;
    if (op == "sqrt") {
      d = div(scalar(0.5f, x.backend()), saved);
    } else if (op == "rsqrt") {
      d = mul(pow(saved, 3), -0.5f);
    } else if (op == "exp") {
      d = saved;
    } else if (op == "log") {
      d = div(scalar(1, x.backend()), x);
    } else if (op == "sin") {
      d = cos(x);
    } else if (op == "cos") {
      d = mul(sin(x), -1);
    } else if (op == "abs") {
      d = sign(x);
    } else if (op == "sign") {
      d = Tensor::zeros(x.shape(), x.backend());
    } else if (op == "relu") {
      d = cast(gt(x, scalar(0, x.backend())), DType::Float32);
    } else if (op == "leaky_relu") {
      d = add(
          mul(cast(gt(x, scalar(0, x.backend())), DType::Float32), 1 - slope),
          slope);
    } else if (op == "silu") {
      Tensor sigmoid = div(scalar(1, x.backend()), add(exp(mul(x, -1)), 1));
      d = mul(sigmoid, add(mul(x, sub(scalar(1, x.backend()), sigmoid)), 1));
    } else if (op == "gelu") {
      Tensor u = mul(add(x, mul(pow(x, 3), 0.044715f)), 0.7978845608028654f);
      Tensor t = sub(div(scalar(2, x.backend()), add(exp(mul(u, -2)), 1)), 1);
      d = add(mul(add(t, 1), 0.5f),
              mul(mul(mul(x, sub(scalar(1, x.backend()), mul(t, t))),
                      add(mul(mul(x, x), 0.134145f), 1)),
                  0.3989422804014327f));
    }
    accumulate(x, mul(g, d));
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
Tensor leaky_relu(const Tensor& x, float slope) {
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
      require(shape[i] > 0, "Invalid reshape dimension");
      known *= shape[i];
    }
  }
  if (inferred >= 0) {
    require(x.numel() % known == 0, "Cannot infer reshape dimension");
    shape[inferred] = int(x.numel() / known);
  }
  require(shape_element_count(shape) == x.numel(),
          "Reshape changes element count");
  auto n = std::make_shared<TensorNode>();
  n->shape = shape;
  n->value = x.node()->value;
  n->operation = "reshape";
  return tracked(Tensor(n), {x}, [x](const Tensor& g) {
    accumulate(x, reshape(g, x.shape()));
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
  require(stop > start, "Empty slices are not supported");
  Shape s = x.shape();
  s[dim] = (stop - start + step - 1) / step;
  return tracked(
      dispatch_tensor("slice", s, {x}, {dim, start, step}, x.dtype()), {x},
      [x, dim, start, step](const Tensor& g) {
        accumulate(x, dispatch_tensor("slice_grad", x.shape(), {g},
                                      {dim, start, step}));
      });
}
static void validate_indices(const Tensor& ids, int count) {
  require(ids.dtype() != DType::Float32,
          "Indices must use int32 or bool dtype");
  for (float v : ids.data()) {
    require(std::isfinite(v) && v >= 0 && v < count && std::floor(v) == v,
            "Index out of range");
  }
}
Tensor gather(const Tensor& x, int dim, const Tensor& ids) {
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
        accumulate(x,
                   dispatch_tensor("gather_grad", x.shape(), {g, ids}, {dim}));
      });
}
Tensor index_select(const Tensor& x, int dim, const Tensor& ids) {
  dim = axis(dim, x.shape().size());
  require(ids.shape().size() == 1, "index_select indices must be a vector");
  validate_indices(ids, x.shape()[dim]);
  Shape s = x.shape();
  s[dim] = int(ids.numel());
  return tracked(dispatch_tensor("index_select", s, {x, ids}, {dim}, x.dtype()),
                 {x}, [x, ids, dim](const Tensor& g) {
                   accumulate(x, dispatch_tensor("index_select_grad", x.shape(),
                                                 {g, ids}, {dim}));
                 });
}
Tensor scatter(const Tensor& x, int dim, const Tensor& ids,
               const Tensor& source) {
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
                     {g, ids, Tensor::zeros(ids.shape(), x.backend())}, {dim}));
        }
        if (source.requires_grad()) {
          accumulate(source, dispatch_tensor("scatter_source_grad",
                                             source.shape(), {g, ids}, {dim}));
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
    require(reps[i] > 0, "repeat factors must be positive");
    s[i] *= reps[i];
  }
  Tensor y = reshape(x, padded);
  return tracked(
      dispatch_tensor("repeat", s, {y}, {}, x.dtype()), {x},
      [x, padded](const Tensor& g) {
        accumulate(
            x, reshape(dispatch_tensor("repeat_grad", padded, {g}), x.shape()));
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
      [x](const Tensor& g) { accumulate(x, unbroadcast(g, x.shape())); });
}
Tensor squeeze(const Tensor& x, int dim) {
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
  int d = all ? -1 : axis(dim, x.shape().size());
  Shape s = x.shape();
  if (all) {
    s = keepdim ? Shape(s.size(), 1) : Shape{};
  } else if (keepdim) {
    s[d] = 1;
  } else {
    s.erase(s.begin() + d);
  }
  Tensor out = dispatch_tensor(op, s, {x}, {d, int(keepdim)}),
         saved = out.detach();
  int kind = op == "mean"                    ? 1
             : (op == "max" || op == "amax") ? 2
             : (op == "min" || op == "amin") ? 3
             : op == "prod"                  ? 4
                                             : 0;
  return tracked(out, {x}, [x, saved, d, keepdim, kind](const Tensor& g) {
    accumulate(x, dispatch_tensor("reduce_grad", x.shape(), {x, saved, g},
                                  {d, int(keepdim), kind}));
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
Tensor Tensor::uniform(Shape shape, float low, float high, uint32_t seed,
                       Backend b, bool grad) {
  require(std::isfinite(low) && std::isfinite(high) && low <= high,
          "Invalid uniform range");
  Tensor t = dispatch_tensor("uniform", shape, {},
                             {int(seed), 0, float_bits(low), float_bits(high)},
                             DType::Float32, b);
  t.node()->requires_grad = grad;
  return t;
}
Tensor Tensor::normal(Shape shape, float mean, float sd, uint32_t seed,
                      Backend b, bool grad) {
  require(std::isfinite(mean) && std::isfinite(sd) && sd >= 0,
          "Invalid normal parameters");
  Tensor t = dispatch_tensor("normal", shape, {},
                             {int(seed), 0, float_bits(mean), float_bits(sd)},
                             DType::Float32, b);
  t.node()->requires_grad = grad;
  return t;
}
Tensor dropout(const Tensor& x, float p, uint32_t seed, bool training) {
  require(p >= 0 && p < 1, "Dropout probability must be in [0,1)");
  if (!training || p == 0) {
    return x;
  }
  return tracked(dispatch_tensor("dropout", x.shape(), {x},
                                 {int(seed), 0, float_bits(p), 1}),
                 {x}, [x, p, seed](const Tensor& g) {
                   accumulate(
                       x, dispatch_tensor("dropout", g.shape(), {g},
                                          {int(seed), 0, float_bits(p), 1}));
                 });
}
Tensor matmul(const Tensor& a, const Tensor& b) {
  require(!a.shape().empty() && !b.shape().empty(),
          "matmul requires rank >= 1");
  if (a.shape().size() == 1 || b.shape().size() == 1) {
    bool av = a.shape().size() == 1, bv = b.shape().size() == 1;
    Tensor y = matmul(av ? unsqueeze(a, 0) : a, bv ? unsqueeze(b, -1) : b);
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
  return tracked(
      dispatch_tensor("matmul", s, {a, b}), {a, b}, [a, b](const Tensor& g) {
        if (a.requires_grad()) {
          accumulate(a,
                     unbroadcast(matmul(g, transpose(b, -2, -1)), a.shape()));
        }
        if (b.requires_grad()) {
          accumulate(b,
                     unbroadcast(matmul(transpose(a, -2, -1), g), b.shape()));
        }
      });
}
Tensor bmm(const Tensor& a, const Tensor& b) {
  require(a.shape().size() == 3 && b.shape().size() == 3 &&
              a.shape()[0] == b.shape()[0],
          "bmm requires equal rank-3 batch sizes");
  return matmul(a, b);
}
Tensor addmm(const Tensor& c, const Tensor& a, const Tensor& b, float beta,
             float alpha) {
  return add(mul(matmul(a, b), alpha), mul(c, beta));
}
static Tensor triangular(const std::string& op, const Tensor& x, int diagonal) {
  require(x.shape().size() >= 2, "Triangular operation requires rank >=2");
  return tracked(dispatch_tensor(op, x.shape(), {x}, {diagonal}, x.dtype()),
                 {x}, [x, diagonal, op](const Tensor& g) {
                   accumulate(x,
                              dispatch_tensor(op, g.shape(), {g}, {diagonal}));
                 });
}
Tensor triu(const Tensor& x, int d) {
  return triangular("triu", x, d);
}
Tensor tril(const Tensor& x, int d) {
  return triangular("tril", x, d);
}
Tensor softmax(const Tensor& x, int dim) {
  dim = axis(dim, x.shape().size());
  Tensor y = dispatch_tensor("softmax", x.shape(), {x}, {dim}),
         saved = y.detach();
  return tracked(y, {x}, [x, saved, dim](const Tensor& g) {
    accumulate(x, mul(saved, sub(g, sum(mul(g, saved), dim, true))));
  });
}
Tensor log_softmax(const Tensor& x, int dim) {
  dim = axis(dim, x.shape().size());
  Tensor y = dispatch_tensor("log_softmax", x.shape(), {x}, {dim}),
         saved = y.detach();
  return tracked(y, {x}, [x, saved, dim](const Tensor& g) {
    accumulate(x, sub(g, mul(exp(saved), sum(g, dim, true))));
  });
}
Tensor layer_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                  float eps) {
  require(eps > 0, "Normalization epsilon must be positive");
  require(!x.shape().empty() && w.shape() == Shape{x.shape().back()} &&
              b.shape() == w.shape(),
          "layer_norm affine shape mismatch");
  Tensor centered = sub(x, mean(x, -1, true));
  return add(mul(mul(centered,
                     rsqrt(add(mean(mul(centered, centered), -1, true), eps))),
                 w),
             b);
}
Tensor rms_norm(const Tensor& x, const Tensor& w, float eps) {
  require(eps > 0 && !x.shape().empty() && w.shape() == Shape{x.shape().back()},
          "Invalid RMS normalization parameters");
  return mul(mul(x, rsqrt(add(mean(mul(x, x), -1, true), eps))), w);
}
Tensor batch_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                  float eps) {
  require(x.shape().size() >= 2 && w.shape() == Shape{x.shape()[1]} &&
              b.shape() == w.shape() && eps > 0,
          "Invalid batch normalization parameters");
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
                  float eps) {
  require(x.shape().size() >= 2 && groups > 0 && x.shape()[1] % groups == 0 &&
              w.shape() == Shape{x.shape()[1]} && b.shape() == w.shape() &&
              eps > 0,
          "Invalid group normalization parameters");
  Tensor c = reshape(x, {x.shape()[0], groups, -1});
  Tensor centered = sub(c, mean(c, -1, true));
  Tensor y = reshape(
      mul(centered, rsqrt(add(mean(mul(centered, centered), -1, true), eps))),
      x.shape());
  Shape affine(x.shape().size(), 1);
  affine[1] = x.shape()[1];
  return add(mul(y, reshape(w, affine)), reshape(b, affine));
}
Tensor mse_loss(const Tensor& a, const Tensor& b) {
  require(a.shape() == b.shape(), "MSE shape mismatch");
  Tensor d = sub(a, b);
  return mean_all(mul(d, d));
}
Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
  require(logits.shape().size() >= 2,
          "Cross entropy requires logits with a class dimension");
  int classes = logits.shape().back();
  require(targets.numel() == logits.numel() / classes,
          "Cross entropy target size mismatch");
  validate_indices(targets, classes);
  Tensor x = reshape(logits, {-1, classes}), y = reshape(targets, {-1});
  Tensor loss = dispatch_tensor("cross_entropy", {}, {x, y});
  return tracked(loss, {x}, [x, y](const Tensor& g) {
    accumulate(x, dispatch_tensor("cross_entropy_grad", x.shape(), {x, y, g}));
  });
}
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride,
              int padding) {
  require(x.shape().size() == 4 && w.shape().size() == 4 &&
              x.shape()[1] == w.shape()[1] && stride > 0 && padding >= 0,
          "Invalid convolution shapes or attributes");
  require(!b.defined() || b.shape() == Shape{w.shape()[0]},
          "Convolution bias shape mismatch");
  int h = (x.shape()[2] + 2 * padding - w.shape()[2]) / stride + 1,
      ww = (x.shape()[3] + 2 * padding - w.shape()[3]) / stride + 1;
  require(x.shape()[2] + 2 * padding >= w.shape()[2] &&
              x.shape()[3] + 2 * padding >= w.shape()[3],
          "Convolution kernel exceeds padded input");
  Shape s{x.shape()[0], w.shape()[0], h, ww};
  std::vector<int> attrs{stride, stride, padding, padding, 1, 1, 1};
  return tracked(dispatch_tensor("conv2d", s, {x, w, b}, attrs), {x, w, b},
                 [x, w, b, attrs](const Tensor& g) {
                   if (x.requires_grad()) {
                     accumulate(x, dispatch_tensor("conv2d_input_grad",
                                                   x.shape(), {g, w}, attrs));
                   }
                   if (w.requires_grad()) {
                     accumulate(w, dispatch_tensor("conv2d_weight_grad",
                                                   w.shape(), {x, g}, attrs));
                   }
                   if (b.requires_grad()) {
                     accumulate(b, sum(sum(sum(g, 3), 2), 0));
                   }
                 });
}
static Tensor pool(const std::string& op, const Tensor& x, int kernel,
                   int stride, int padding) {
  if (stride == 0) {
    stride = kernel;
  }
  require(x.shape().size() == 4 && kernel > 0 && stride > 0 && padding >= 0 &&
              padding <= kernel / 2,
          "Invalid pooling parameters");
  require(x.shape()[2] + 2 * padding >= kernel &&
              x.shape()[3] + 2 * padding >= kernel,
          "Pooling kernel exceeds padded input");
  Shape s = x.shape();
  s[2] = (s[2] + 2 * padding - kernel) / stride + 1;
  s[3] = (s[3] + 2 * padding - kernel) / stride + 1;
  std::vector<int> attrs{kernel, kernel, stride, stride, padding, padding, 1};
  return tracked(
      dispatch_tensor(op, s, {x}, attrs), {x}, [x, attrs, op](const Tensor& g) {
        accumulate(x, dispatch_tensor(op + "_grad", x.shape(), {x, g}, attrs));
      });
}
Tensor max_pool2d(const Tensor& x, int k, int s, int p) {
  return pool("max_pool2d", x, k, s, p);
}
Tensor avg_pool2d(const Tensor& x, int k, int s, int p) {
  return pool("avg_pool2d", x, k, s, p);
}
}  // namespace elman
