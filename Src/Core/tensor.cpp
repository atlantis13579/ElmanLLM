#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace elman {
struct TensorNode {
    Shape shape;
    BufferPtr value;
    bool requires_grad = false;
    std::shared_ptr<TensorNode> gradient;
    std::vector<std::shared_ptr<TensorNode>> parents;
    std::vector<uint64_t> versions;
    std::function<void(const Tensor&)> backward;
    std::string operation = "leaf";
    uint64_t saved_version = 0;
};
static thread_local bool recording = true;
struct NoGrad {
    bool old = recording;
    NoGrad() { recording = false; }
    ~NoGrad() { recording = old; }
};
static void require(bool ok, const std::string& message) {
    if (!ok)
        throw std::invalid_argument(message);
}
static size_t elements(const Shape& shape) {
    require(shape.size() <= 8, "Maximum tensor rank is eight");
    size_t n = 1;
    for (int d : shape) {
        require(d > 0, "Tensor dimensions must be positive");
        require(n <= size_t(std::numeric_limits<int>::max()) / size_t(d), "Tensor exceeds 32-bit indexing limit");
        n *= size_t(d);
    }
    return n;
}
static int axis(int dim, size_t rank) {
    if (dim < 0)
        dim += int(rank);
    require(dim >= 0 && dim < int(rank), "Axis is out of range");
    return dim;
}
static void require_defined(const Tensor& x) {
    require(x.defined(), "Undefined tensor");
}
const Shape& Tensor::shape() const {
    require_defined(*this);
    return node_->shape;
}
size_t Tensor::numel() const {
    require_defined(*this);
    return node_->value->count;
}
Backend Tensor::backend() const {
    require_defined(*this);
    return node_->value->backend;
}
DType Tensor::dtype() const {
    require_defined(*this);
    return node_->value->dtype;
}
bool Tensor::requires_grad() const {
    return node_ && node_->requires_grad;
}
std::vector<float> Tensor::data() const {
    require_defined(*this);
    return backend_download(node_->value);
}
std::vector<float> Tensor::grad() const {
    require_defined(*this);
    return node_->gradient ? Tensor(node_->gradient).data() : std::vector<float>{};
}
void Tensor::zero_grad() {
    require_defined(*this);
    node_->gradient.reset();
}
void Tensor::assign(const std::vector<float>& values) {
    require_defined(*this);
    backend_upload(node_->value, values);
    ++node_->value->version;
}
Tensor Tensor::from_data(Shape shape, const std::vector<float>& data, Backend b, bool grad, DType dtype) {
    require(elements(shape) == data.size(), "Data and tensor shape sizes differ");
    require(!grad || dtype == DType::Float32, "Only float32 tensors can require gradients");
    auto n = std::make_shared<TensorNode>();
    n->shape = std::move(shape);
    n->value = backend_allocate(b, data.size(), dtype);
    n->requires_grad = grad;
    backend_upload(n->value, data);
    return Tensor(n);
}
Tensor Tensor::zeros(Shape s, Backend b, bool grad) {
    return from_data(s, std::vector<float>(elements(s), 0), b, grad);
}
Tensor Tensor::ones(Shape s, Backend b, bool grad) {
    return from_data(s, std::vector<float>(elements(s), 1), b, grad);
}
Tensor Tensor::detach() const {
    require_defined(*this);
    auto n = std::make_shared<TensorNode>();
    n->shape = shape();
    n->value = node_->value;
    return Tensor(n);
}
Tensor Tensor::to(Backend b) const {
    return from_data(shape(), data(), b, requires_grad(), dtype());
}
static int float_bits(float v) {
    int n;
    std::memcpy(&n, &v, 4);
    return n;
}
static Tensor raw(const std::string& op, const Shape& shape, const std::vector<Tensor>& inputs,
                  const std::vector<int>& attrs = {}, DType dtype = DType::Float32, Backend backend = Backend::CPU) {
    auto n = std::make_shared<TensorNode>();
    n->shape = shape;
    if (!inputs.empty())
        backend = inputs[0].backend();
    n->value = backend_allocate(backend, elements(shape), dtype);
    n->operation = op;
    KernelParams p;
    p.v[0] = int(elements(shape));
    p.v[1] = int(shape.size());
    std::copy(shape.begin(), shape.end(), p.v + 2);
    require(inputs.size() <= 3, "Kernel accepts at most three inputs");
    std::vector<BufferPtr> buffers;
    for (int k = 0; k < 3; ++k) {
        int base = 10 + 17 * k;
        p.v[base] = -1;
        if (k >= int(inputs.size()) || !inputs[k].defined()) {
            buffers.push_back(nullptr);
            continue;
        }
        const Tensor& t = inputs[k];
        require(t.backend() == backend, "Mixed-backend operation");
        p.v[base] = int(t.shape().size());
        std::copy(t.shape().begin(), t.shape().end(), p.v + base + 1);
        int stride = 1;
        for (int d = int(t.shape().size()) - 1; d >= 0; --d) {
            p.v[base + 9 + d] = stride;
            stride *= t.shape()[d];
        }
        p.v[252 + k] = int(t.dtype());
        buffers.push_back(t.node()->value);
    }
    require(attrs.size() <= 188, "Too many kernel attributes");
    std::copy(attrs.begin(), attrs.end(), p.v + 64);
    p.v[255] = int(dtype);
    backend_dispatch(op, n->value, buffers, p);
    return Tensor(n);
}
static Tensor tracked(Tensor out, const std::vector<Tensor>& parents, std::function<void(const Tensor&)> backward) {
    if (!recording)
        return out;
    bool needed = false;
    for (const auto& p : parents)
        needed |= p.requires_grad();
    if (!needed)
        return out;
    out.node()->requires_grad = true;
    out.node()->backward = std::move(backward);
    out.node()->saved_version = out.node()->value->version;
    for (const auto& p : parents)
        if (p.defined()) {
            out.node()->parents.push_back(p.node());
            out.node()->versions.push_back(p.node()->value->version);
        }
    return out;
}
static void accumulate(const Tensor& t, const Tensor& g) {
    if (!t.requires_grad())
        return;
    require(t.shape() == g.shape(), "Gradient shape mismatch");
    auto& current = t.node()->gradient;
    current = current ? add(Tensor(current), g).node() : g.detach().node();
}
void Tensor::backward() {
    require(numel() == 1, "Non-scalar backward requires an explicit gradient");
    backward(ones(shape(), backend()));
}
void Tensor::backward(const Tensor& seed) {
    require(requires_grad(), "Tensor does not require gradients");
    require(seed.shape() == shape() && seed.backend() == backend(), "Backward seed mismatch");
    NoGrad no_grad;
    std::vector<std::shared_ptr<TensorNode>> topo;
    std::unordered_set<TensorNode*> seen;
    std::function<void(const std::shared_ptr<TensorNode>&)> visit = [&](const auto& n) {
        if (!seen.insert(n.get()).second)
            return;
        require(!n->backward || n->saved_version == n->value->version,
                "A graph output was modified before backward");
        for (size_t i = 0; i < n->parents.size(); ++i) {
            require(n->versions[i] == n->parents[i]->value->version, "A graph input was modified before backward");
            visit(n->parents[i]);
        }
        topo.push_back(n);
    };
    visit(node_);
    for (auto& n : topo)
        if (!n->parents.empty())
            n->gradient.reset();
    accumulate(*this, seed);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if ((*it)->gradient && (*it)->backward)
            (*it)->backward(Tensor((*it)->gradient));
}
std::string Tensor::graph_json() const {
    require_defined(*this);
    std::unordered_map<TensorNode*, size_t> ids;
    std::vector<std::shared_ptr<TensorNode>> nodes;
    std::function<void(const std::shared_ptr<TensorNode>&)> visit = [&](const auto& n) {
        if (ids.count(n.get()))
            return;
        for (auto& p : n->parents)
            visit(p);
        ids[n.get()] = nodes.size();
        nodes.push_back(n);
    };
    visit(node_);
    std::ostringstream s;
    s << "{\"format\":\"elman.graph\",\"version\":1,\"nodes\":[";
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto n = nodes[i];
        if (i)
            s << ',';
        s << "{\"id\":" << i << ",\"op\":\"" << n->operation << "\",\"shape\":[";
        for (size_t j = 0; j < n->shape.size(); ++j) {
            if (j)
                s << ',';
            s << n->shape[j];
        }
        s << "],\"requires_grad\":" << (n->requires_grad ? "true" : "false") << ",\"inputs\":[";
        for (size_t j = 0; j < n->parents.size(); ++j) {
            if (j)
                s << ',';
            s << ids[n->parents[j].get()];
        }
        s << "]}";
    }
    s << "],\"output\":" << ids[node_.get()] << '}';
    return s.str();
}
static Shape broadcast(const Shape& a, const Shape& b) {
    Shape out(std::max(a.size(), b.size()), 1);
    for (size_t k = 0; k < out.size(); ++k) {
        int x = k < a.size() ? a[a.size() - 1 - k] : 1, y = k < b.size() ? b[b.size() - 1 - k] : 1;
        require(x == y || x == 1 || y == 1, "Shapes cannot broadcast");
        out[out.size() - 1 - k] = std::max(x, y);
    }
    return out;
}
static Tensor unbroadcast(Tensor g, const Shape& shape) {
    while (g.shape().size() > shape.size())
        g = sum(g, 0);
    for (int d = 0; d < int(shape.size()); ++d)
        if (shape[d] == 1 && g.shape()[d] != 1)
            g = sum(g, d, true);
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
    Tensor out = raw(op, s, {a, b});
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
            if (a.requires_grad())
                ga = mul(g, mul(b, pow(a, sub(b, 1))));
            if (b.requires_grad())
                gb = mul(g, mul(pow(a, b), log(a)));
        } else {
            bool mx = op == "maximum";
            Tensor tie = mul(cast(eq(a, b), DType::Float32), 0.5f);
            ga = mul(g, add(cast(mx ? gt(a, b) : lt(a, b), DType::Float32), tie));
            gb = mul(g, add(cast(mx ? lt(a, b) : gt(a, b), DType::Float32), tie));
        }
        if (a.requires_grad())
            accumulate(a, unbroadcast(ga, a.shape()));
        if (b.requires_grad())
            accumulate(b, unbroadcast(gb, b.shape()));
    });
}
#define BINARY(name)                                                                                                   \
    Tensor name(const Tensor& a, const Tensor& b) {                                                                    \
        return binary(#name, a, b);                                                                                    \
    }
BINARY(add)
BINARY(sub) BINARY(mul) BINARY(div) BINARY(pow) BINARY(maximum) BINARY(minimum)
#undef BINARY
#define COMPARE(name)                                                                                                  \
    Tensor name(const Tensor& a, const Tensor& b) {                                                                    \
        return raw(#name, broadcast(a.shape(), b.shape()), {a, b}, {}, DType::Bool);                                   \
    }
    COMPARE(eq) COMPARE(ne) COMPARE(gt) COMPARE(ge) COMPARE(lt) COMPARE(le) COMPARE(logical_and) COMPARE(logical_or)
#undef COMPARE
        Tensor logical_not(const Tensor& x) {
    return raw("logical_not", x.shape(), {x}, {}, DType::Bool);
}
Tensor cast(const Tensor& x, DType dtype) {
    Tensor y = raw("cast", x.shape(), {x}, {}, dtype);
    if (x.dtype() == DType::Float32 && dtype == DType::Float32)
        return tracked(y, {x}, [x](const Tensor& g) { accumulate(x, g); });
    return y;
}
static Tensor unary(const std::string& op, const Tensor& x, float slope = 0.01f) {
    Tensor out = raw(op, x.shape(), {x}, {float_bits(slope)}), saved = out.detach();
    return tracked(out, {x}, [x, saved, op, slope](const Tensor& g) {
        Tensor d;
        if (op == "sqrt")
            d = div(scalar(0.5f, x.backend()), saved);
        else if (op == "rsqrt")
            d = mul(pow(saved, 3), -0.5f);
        else if (op == "exp")
            d = saved;
        else if (op == "log")
            d = div(scalar(1, x.backend()), x);
        else if (op == "sin")
            d = cos(x);
        else if (op == "cos")
            d = mul(sin(x), -1);
        else if (op == "abs")
            d = sign(x);
        else if (op == "sign")
            d = Tensor::zeros(x.shape(), x.backend());
        else if (op == "relu")
            d = cast(gt(x, scalar(0, x.backend())), DType::Float32);
        else if (op == "leaky_relu")
            d = add(mul(cast(gt(x, scalar(0, x.backend())), DType::Float32), 1 - slope), slope);
        else if (op == "silu") {
            Tensor sigmoid = div(scalar(1, x.backend()), add(exp(mul(x, -1)), 1));
            d = mul(sigmoid, add(mul(x, sub(scalar(1, x.backend()), sigmoid)), 1));
        } else if (op == "gelu") {
            Tensor u = mul(add(x, mul(pow(x, 3), 0.044715f)), 0.7978845608028654f);
            Tensor t = sub(div(scalar(2, x.backend()), add(exp(mul(u, -2)), 1)), 1);
            d = add(mul(add(t, 1), 0.5f),
                    mul(mul(mul(x, sub(scalar(1, x.backend()), mul(t, t))), add(mul(mul(x, x), 0.134145f), 1)),
                        0.3989422804014327f));
        }
        accumulate(x, mul(g, d));
    });
}
#define UNARY(name)                                                                                                    \
    Tensor name(const Tensor& x) {                                                                                     \
        return unary(#name, x);                                                                                        \
    }
UNARY(sqrt)
UNARY(rsqrt) UNARY(exp) UNARY(log) UNARY(sin) UNARY(cos) UNARY(abs) UNARY(sign) UNARY(relu) UNARY(silu) UNARY(gelu)
#undef UNARY
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
    require(elements(shape) == x.numel(), "Reshape changes element count");
    auto n = std::make_shared<TensorNode>();
    n->shape = shape;
    n->value = x.node()->value;
    n->operation = "reshape";
    return tracked(Tensor(n), {x}, [x](const Tensor& g) { accumulate(x, reshape(g, x.shape())); });
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
    return tracked(raw("permute", s, {x}, normalized, x.dtype()), {x},
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
    if (start < 0)
        start += n;
    if (stop < 0)
        stop += n;
    start = std::max(0, std::min(n, start));
    stop = std::max(0, std::min(n, stop));
    require(stop > start, "Empty slices are not supported");
    Shape s = x.shape();
    s[dim] = (stop - start + step - 1) / step;
    return tracked(raw("slice", s, {x}, {dim, start, step}, x.dtype()), {x}, [x, dim, start, step](const Tensor& g) {
        accumulate(x, raw("slice_grad", x.shape(), {g}, {dim, start, step}));
    });
}
static void validate_indices(const Tensor& ids, int count) {
    require(ids.dtype() != DType::Float32, "Indices must use int32 or bool dtype");
    for (float v : ids.data())
        require(std::isfinite(v) && v >= 0 && v < count && std::floor(v) == v, "Index out of range");
}
Tensor gather(const Tensor& x, int dim, const Tensor& ids) {
    dim = axis(dim, x.shape().size());
    require(ids.shape().size() == x.shape().size(), "Gather rank mismatch");
    for (int j = 0; j < int(x.shape().size()); ++j)
        if (j != dim)
            require(ids.shape()[j] <= x.shape()[j], "Gather dimension exceeds input");
    validate_indices(ids, x.shape()[dim]);
    return tracked(raw("gather", ids.shape(), {x, ids}, {dim}, x.dtype()), {x},
                   [x, ids, dim](const Tensor& g) { accumulate(x, raw("gather_grad", x.shape(), {g, ids}, {dim})); });
}
Tensor index_select(const Tensor& x, int dim, const Tensor& ids) {
    dim = axis(dim, x.shape().size());
    require(ids.shape().size() == 1, "index_select indices must be a vector");
    validate_indices(ids, x.shape()[dim]);
    Shape s = x.shape();
    s[dim] = int(ids.numel());
    return tracked(raw("index_select", s, {x, ids}, {dim}, x.dtype()), {x}, [x, ids, dim](const Tensor& g) {
        accumulate(x, raw("index_select_grad", x.shape(), {g, ids}, {dim}));
    });
}
Tensor scatter(const Tensor& x, int dim, const Tensor& ids, const Tensor& source) {
    dim = axis(dim, x.shape().size());
    require(source.shape() == ids.shape() && ids.shape().size() == x.shape().size(),
            "Scatter source/index shape mismatch");
    for (int d = 0; d < int(x.shape().size()); ++d)
        if (d != dim)
            require(ids.shape()[d] <= x.shape()[d], "Scatter shape exceeds destination");
    validate_indices(ids, x.shape()[dim]);
    return tracked(raw("scatter", x.shape(), {x, ids, source}, {dim}, x.dtype()), {x, source},
                   [x, ids, source, dim](const Tensor& g) {
                       if (x.requires_grad())
                           accumulate(
                               x, raw("scatter", x.shape(), {g, ids, Tensor::zeros(ids.shape(), x.backend())}, {dim}));
                       if (source.requires_grad())
                           accumulate(source, raw("scatter_source_grad", source.shape(), {g, ids}, {dim}));
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
        for (int d = 0; d < int(s.size()); ++d)
            if (d != dim)
                require(s[d] == b.shape()[d], "concat dimension mismatch");
        int boundary = s[dim];
        s[dim] += b.shape()[dim];
        result = tracked(raw("concat", s, {a, b}, {dim}, a.dtype()), {a, b}, [a, b, dim, boundary](const Tensor& g) {
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
    for (int start = 0; start < x.shape()[dim]; start += size)
        out.push_back(slice(x, dim, start, std::min(start + size, x.shape()[dim])));
    return out;
}
Tensor repeat(const Tensor& x, const Shape& reps) {
    require(reps.size() >= x.shape().size(), "repeat rank is smaller than input rank");
    Shape padded(reps.size() - x.shape().size(), 1);
    padded.insert(padded.end(), x.shape().begin(), x.shape().end());
    Shape s = padded;
    for (size_t i = 0; i < reps.size(); ++i) {
        require(reps[i] > 0, "repeat factors must be positive");
        s[i] *= reps[i];
    }
    Tensor y = reshape(x, padded);
    return tracked(raw("repeat", s, {y}, {}, x.dtype()), {x}, [x, padded](const Tensor& g) {
        accumulate(x, reshape(raw("repeat_grad", padded, {g}), x.shape()));
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
    return tracked(raw("expand", s, {x}, {}, x.dtype()), {x},
                   [x](const Tensor& g) { accumulate(x, unbroadcast(g, x.shape())); });
}
Tensor squeeze(const Tensor& x, int dim) {
    dim = axis(dim, x.shape().size());
    Shape s = x.shape();
    if (s[dim] == 1)
        s.erase(s.begin() + dim);
    return reshape(x, s);
}
Tensor squeeze(const Tensor& x) {
    Shape s;
    for (int d : x.shape())
        if (d != 1)
            s.push_back(d);
    return reshape(x, s);
}
Tensor unsqueeze(const Tensor& x, int dim) {
    if (dim < 0)
        dim += int(x.shape().size()) + 1;
    require(dim >= 0 && dim <= int(x.shape().size()), "unsqueeze axis out of range");
    Shape s = x.shape();
    s.insert(s.begin() + dim, 1);
    return reshape(x, s);
}
static Tensor reduce(const std::string& op, const Tensor& x, int dim, bool keepdim, bool all = false) {
    int d = all ? -1 : axis(dim, x.shape().size());
    Shape s = x.shape();
    if (all)
        s = keepdim ? Shape(s.size(), 1) : Shape{};
    else if (keepdim)
        s[d] = 1;
    else
        s.erase(s.begin() + d);
    Tensor out = raw(op, s, {x}, {d, int(keepdim)}), saved = out.detach();
    int kind = op == "mean"                    ? 1
               : (op == "max" || op == "amax") ? 2
               : (op == "min" || op == "amin") ? 3
               : op == "prod"                  ? 4
                                               : 0;
    return tracked(out, {x}, [x, saved, d, keepdim, kind](const Tensor& g) {
        accumulate(x, raw("reduce_grad", x.shape(), {x, saved, g}, {d, int(keepdim), kind}));
    });
}
#define REDUCE(name)                                                                                                   \
    Tensor name(const Tensor& x, int d, bool k) {                                                                      \
        return reduce(#name, x, d, k);                                                                                 \
    }
REDUCE(sum)
REDUCE(mean) REDUCE(max) REDUCE(min) REDUCE(amax) REDUCE(amin) REDUCE(prod)
#undef REDUCE
    Tensor sum_all(const Tensor& x) {
    return reduce("sum", x, 0, false, true);
}
Tensor mean_all(const Tensor& x) {
    return reduce("mean", x, 0, false, true);
}
Tensor Tensor::uniform(Shape shape, float low, float high, uint32_t seed, Backend b, bool grad) {
    require(std::isfinite(low) && std::isfinite(high) && low <= high, "Invalid uniform range");
    Tensor t = raw("uniform", shape, {}, {int(seed), 0, float_bits(low), float_bits(high)}, DType::Float32, b);
    t.node()->requires_grad = grad;
    return t;
}
Tensor Tensor::normal(Shape shape, float mean, float sd, uint32_t seed, Backend b, bool grad) {
    require(std::isfinite(mean) && std::isfinite(sd) && sd >= 0, "Invalid normal parameters");
    Tensor t = raw("normal", shape, {}, {int(seed), 0, float_bits(mean), float_bits(sd)}, DType::Float32, b);
    t.node()->requires_grad = grad;
    return t;
}
Tensor dropout(const Tensor& x, float p, uint32_t seed, bool training) {
    require(p >= 0 && p < 1, "Dropout probability must be in [0,1)");
    if (!training || p == 0)
        return x;
    return tracked(raw("dropout", x.shape(), {x}, {int(seed), 0, float_bits(p), 1}), {x},
                   [x, p, seed](const Tensor& g) {
                       accumulate(x, raw("dropout", g.shape(), {g}, {int(seed), 0, float_bits(p), 1}));
                   });
}
Tensor matmul(const Tensor& a, const Tensor& b) {
    require(!a.shape().empty() && !b.shape().empty(), "matmul requires rank >= 1");
    if (a.shape().size() == 1 || b.shape().size() == 1) {
        bool av = a.shape().size() == 1, bv = b.shape().size() == 1;
        Tensor y = matmul(av ? unsqueeze(a, 0) : a, bv ? unsqueeze(b, -1) : b);
        if (av)
            y = squeeze(y, int(y.shape().size()) - 2);
        if (bv)
            y = squeeze(y, -1);
        return y;
    }
    int ar = int(a.shape().size()), br = int(b.shape().size());
    require(a.shape()[ar - 1] == b.shape()[br - 2], "matmul inner dimension mismatch");
    Shape s = broadcast(Shape(a.shape().begin(), a.shape().end() - 2), Shape(b.shape().begin(), b.shape().end() - 2));
    s.push_back(a.shape()[ar - 2]);
    s.push_back(b.shape()[br - 1]);
    return tracked(raw("matmul", s, {a, b}), {a, b}, [a, b](const Tensor& g) {
        if (a.requires_grad())
            accumulate(a, unbroadcast(matmul(g, transpose(b, -2, -1)), a.shape()));
        if (b.requires_grad())
            accumulate(b, unbroadcast(matmul(transpose(a, -2, -1), g), b.shape()));
    });
}
Tensor bmm(const Tensor& a, const Tensor& b) {
    require(a.shape().size() == 3 && b.shape().size() == 3 && a.shape()[0] == b.shape()[0],
            "bmm requires equal rank-3 batch sizes");
    return matmul(a, b);
}
Tensor addmm(const Tensor& c, const Tensor& a, const Tensor& b, float beta, float alpha) {
    return add(mul(matmul(a, b), alpha), mul(c, beta));
}
static Tensor triangular(const std::string& op, const Tensor& x, int diagonal) {
    require(x.shape().size() >= 2, "Triangular operation requires rank >=2");
    return tracked(raw(op, x.shape(), {x}, {diagonal}, x.dtype()), {x},
                   [x, diagonal, op](const Tensor& g) { accumulate(x, raw(op, g.shape(), {g}, {diagonal})); });
}
Tensor triu(const Tensor& x, int d) {
    return triangular("triu", x, d);
}
Tensor tril(const Tensor& x, int d) {
    return triangular("tril", x, d);
}
Tensor softmax(const Tensor& x, int dim) {
    dim = axis(dim, x.shape().size());
    Tensor y = raw("softmax", x.shape(), {x}, {dim}), saved = y.detach();
    return tracked(
        y, {x}, [x, saved, dim](const Tensor& g) { accumulate(x, mul(saved, sub(g, sum(mul(g, saved), dim, true)))); });
}
Tensor log_softmax(const Tensor& x, int dim) {
    dim = axis(dim, x.shape().size());
    Tensor y = raw("log_softmax", x.shape(), {x}, {dim}), saved = y.detach();
    return tracked(y, {x},
                   [x, saved, dim](const Tensor& g) { accumulate(x, sub(g, mul(exp(saved), sum(g, dim, true)))); });
}
Tensor layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, float eps) {
    require(eps > 0, "Normalization epsilon must be positive");
    require(!x.shape().empty() && w.shape() == Shape{x.shape().back()} && b.shape() == w.shape(),
            "layer_norm affine shape mismatch");
    Tensor centered = sub(x, mean(x, -1, true));
    return add(mul(mul(centered, rsqrt(add(mean(mul(centered, centered), -1, true), eps))), w), b);
}
Tensor rms_norm(const Tensor& x, const Tensor& w, float eps) {
    require(eps > 0 && !x.shape().empty() && w.shape() == Shape{x.shape().back()},
            "Invalid RMS normalization parameters");
    return mul(mul(x, rsqrt(add(mean(mul(x, x), -1, true), eps))), w);
}
Tensor batch_norm(const Tensor& x, const Tensor& w, const Tensor& b, float eps) {
    require(x.shape().size() >= 2 && w.shape() == Shape{x.shape()[1]} && b.shape() == w.shape() && eps > 0,
            "Invalid batch normalization parameters");
    std::vector<int> axes(x.shape().size());
    std::iota(axes.begin(), axes.end(), 0);
    std::swap(axes[0], axes[1]);
    Tensor c = reshape(permute(x, axes), {x.shape()[1], -1});
    Tensor centered = sub(c, mean(c, -1, true));
    Tensor y = add(mul(mul(centered, rsqrt(add(mean(mul(centered, centered), -1, true), eps))), unsqueeze(w, 1)),
                   unsqueeze(b, 1));
    Shape s = x.shape();
    std::swap(s[0], s[1]);
    return permute(reshape(y, s), axes);
}
Tensor group_norm(const Tensor& x, int groups, const Tensor& w, const Tensor& b, float eps) {
    require(x.shape().size() >= 2 && groups > 0 && x.shape()[1] % groups == 0 && w.shape() == Shape{x.shape()[1]} &&
                b.shape() == w.shape() && eps > 0,
            "Invalid group normalization parameters");
    Tensor c = reshape(x, {x.shape()[0], groups, -1});
    Tensor centered = sub(c, mean(c, -1, true));
    Tensor y = reshape(mul(centered, rsqrt(add(mean(mul(centered, centered), -1, true), eps))), x.shape());
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
    require(logits.shape().size() >= 2, "Cross entropy requires logits with a class dimension");
    int classes = logits.shape().back();
    require(targets.numel() == logits.numel() / classes, "Cross entropy target size mismatch");
    validate_indices(targets, classes);
    Tensor x = reshape(logits, {-1, classes}), y = reshape(targets, {-1});
    Tensor loss = raw("cross_entropy", {}, {x, y});
    return tracked(loss, {x},
                   [x, y](const Tensor& g) { accumulate(x, raw("cross_entropy_grad", x.shape(), {x, y, g})); });
}
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride, int padding) {
    require(x.shape().size() == 4 && w.shape().size() == 4 && x.shape()[1] == w.shape()[1] && stride > 0 &&
                padding >= 0,
            "Invalid convolution shapes or attributes");
    require(!b.defined() || b.shape() == Shape{w.shape()[0]}, "Convolution bias shape mismatch");
    int h = (x.shape()[2] + 2 * padding - w.shape()[2]) / stride + 1,
        ww = (x.shape()[3] + 2 * padding - w.shape()[3]) / stride + 1;
    require(x.shape()[2] + 2 * padding >= w.shape()[2] && x.shape()[3] + 2 * padding >= w.shape()[3],
            "Convolution kernel exceeds padded input");
    Shape s{x.shape()[0], w.shape()[0], h, ww};
    std::vector<int> attrs{stride, stride, padding, padding, 1, 1, 1};
    return tracked(raw("conv2d", s, {x, w, b}, attrs), {x, w, b}, [x, w, b, attrs](const Tensor& g) {
        if (x.requires_grad())
            accumulate(x, raw("conv2d_input_grad", x.shape(), {g, w}, attrs));
        if (w.requires_grad())
            accumulate(w, raw("conv2d_weight_grad", w.shape(), {x, g}, attrs));
        if (b.requires_grad())
            accumulate(b, sum(sum(sum(g, 3), 2), 0));
    });
}
static Tensor pool(const std::string& op, const Tensor& x, int kernel, int stride, int padding) {
    if (stride == 0)
        stride = kernel;
    require(x.shape().size() == 4 && kernel > 0 && stride > 0 && padding >= 0 && padding <= kernel / 2,
            "Invalid pooling parameters");
    require(x.shape()[2] + 2 * padding >= kernel && x.shape()[3] + 2 * padding >= kernel,
            "Pooling kernel exceeds padded input");
    Shape s = x.shape();
    s[2] = (s[2] + 2 * padding - kernel) / stride + 1;
    s[3] = (s[3] + 2 * padding - kernel) / stride + 1;
    std::vector<int> attrs{kernel, kernel, stride, stride, padding, padding, 1};
    return tracked(raw(op, s, {x}, attrs), {x},
                   [x, attrs, op](const Tensor& g) { accumulate(x, raw(op + "_grad", x.shape(), {x, g}, attrs)); });
}
Tensor max_pool2d(const Tensor& x, int k, int s, int p) {
    return pool("max_pool2d", x, k, s, p);
}
Tensor avg_pool2d(const Tensor& x, int k, int s, int p) {
    return pool("avg_pool2d", x, k, s, p);
}
} // namespace elman
