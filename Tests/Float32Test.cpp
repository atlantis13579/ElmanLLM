#include "Core/Tensor.h"
#include "IO/Json.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace elman;
namespace {
Backend device;
int checks = 0;
Tensor t(Shape shape, std::vector<float> values, bool grad = false) {
    return Tensor::from_data(shape, values, device, grad);
}
Tensor ids(Shape shape, std::vector<float> values) {
    return Tensor::from_data(shape, values, device, false, DType::Int32);
}
void check(bool ok, const std::string& name) {
    if (!ok)
        throw std::runtime_error(name);
    ++checks;
}
void near(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name,
          float tolerance = 2e-5f) {
    check(actual.size() == expected.size(), name + " size");
    for (size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance * (1 + std::abs(expected[i])))
            throw std::runtime_error(name + " index=" + std::to_string(i) + " actual=" + std::to_string(actual[i]) +
                                     " expected=" + std::to_string(expected[i]));
    ++checks;
}
void expect_throw(const std::function<void()>& fn, const std::string& name) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, name);
}
void gradient(const std::string& name, const Shape& shape, const std::vector<float>& values,
              const std::function<Tensor(const Tensor&)>& fn, float tolerance = 0.008f) {
    Tensor x = t(shape, values, true);
    Tensor loss = sum_all(fn(x));
    loss.backward();
    std::vector<float> analytical = x.grad();
    std::vector<float> numerical(values.size());
    const float delta = 0.001f;
    for (size_t i = 0; i < values.size(); ++i) {
        std::vector<float> plus = values, minus = values;
        plus[i] += delta;
        minus[i] -= delta;
        float hi = sum_all(fn(t(shape, plus))).data()[0], lo = sum_all(fn(t(shape, minus))).data()[0];
        numerical[i] = (hi - lo) / (2 * delta);
    }
    near(analytical, numerical, name + " gradient", tolerance);
}
void arithmetic() {
    Tensor a = t({2, 3}, {1, 2, 3, 4, 5, 6}), b = t({3}, {2, 3, 4});
    near(add(a, b).data(), {3, 5, 7, 6, 8, 10}, "broadcast add");
    near(sub(a, b).data(), {-1, -1, -1, 2, 2, 2}, "sub");
    near(mul(a, b).data(), {2, 6, 12, 8, 15, 24}, "mul");
    near(div(a, b).data(), {.5f, 2.f / 3, .75f, 2, 5.f / 3, 1.5f}, "div");
    near(pow(b, 2).data(), {4, 9, 16}, "pow");
    near(pow(t({4}, {-2, -1, 0, 2}), 3).data(), {-8, -1, 0, 8}, "negative integer power");
    near(pow(t({3}, {-2, -1, 2}), -2).data(), {.25f, 1, .25f}, "negative integer exponent");
    near(pow(t({2}, {0, -2}), 0).data(), {1, 1}, "zero exponent");
    near(sqrt(t({3}, {1, 4, 9})).data(), {1, 2, 3}, "sqrt");
    near(rsqrt(t({3}, {1, 4, 16})).data(), {1, .5f, .25f}, "rsqrt");
    near(log(exp(b)).data(), b.data(), "exp log");
    near(abs(t({3}, {-2, 0, 3})).data(), {2, 0, 3}, "abs");
    near(sign(t({3}, {-2, 0, 3})).data(), {-1, 0, 1}, "sign");
    near(maximum(a, b).data(), {2, 3, 4, 4, 5, 6}, "maximum");
    near(minimum(a, b).data(), {1, 2, 3, 2, 3, 4}, "minimum");
    near(eq(a, b).data(), {0, 0, 0, 0, 0, 0}, "eq");
    near(ne(a, b).data(), {1, 1, 1, 1, 1, 1}, "ne");
    near(gt(a, b).data(), {0, 0, 0, 1, 1, 1}, "gt");
    near(ge(a, b).data(), {0, 0, 0, 1, 1, 1}, "ge");
    near(lt(a, b).data(), {1, 1, 1, 0, 0, 0}, "lt");
    near(le(a, b).data(), {1, 1, 1, 0, 0, 0}, "le");
    Tensor mask = gt(a, b);
    near(logical_not(mask).data(), {1, 1, 1, 0, 0, 0}, "not");
    near(logical_and(mask, logical_not(mask)).data(), {0, 0, 0, 0, 0, 0}, "and");
    near(logical_or(mask, logical_not(mask)).data(), {1, 1, 1, 1, 1, 1}, "or");
    near(cast(t({3}, {1.9f, -2.7f, 0}), DType::Int32).data(), {1, -2, 0}, "cast int");
    near(cast(t({3}, {1.9f, -2.7f, 0}), DType::Bool).data(), {1, 1, 0}, "cast bool");
    for (const std::string op : {"sqrt", "rsqrt", "exp", "log", "sin", "cos", "abs", "silu", "gelu", "leaky_relu"}) {
        gradient(op, {3}, {.3f, .8f, 1.3f}, [op](const Tensor& x) {
            if (op == "sqrt")
                return sqrt(x);
            if (op == "rsqrt")
                return rsqrt(x);
            if (op == "exp")
                return exp(x);
            if (op == "log")
                return log(x);
            if (op == "sin")
                return sin(x);
            if (op == "cos")
                return cos(x);
            if (op == "abs")
                return abs(x);
            if (op == "silu")
                return silu(x);
            if (op == "gelu")
                return gelu(x);
            return leaky_relu(x);
        });
    }
    gradient("relu", {3}, {-.7f, .5f, 1.2f}, [](const Tensor& x) { return relu(x); });
    gradient("gelu negative", {3}, {-1.3f, -.4f, .7f}, [](const Tensor& x) { return gelu(x); });
    gradient("broadcast chain", {1, 3}, {.7f, 1.1f, 1.6f},
             [a](const Tensor& x) { return div(mul(add(a, x), pow(x, 2)), x); });
}
void reductions_and_layout() {
    Tensor a = t({2, 3}, {1, 2, 3, 4, 5, 6});
    near(sum(a, 0).data(), {5, 7, 9}, "sum dim0");
    near(mean(a, 1, true).data(), {2, 5}, "mean keepdim");
    check(mean(a, 1, true).shape() == Shape({2, 1}), "keepdim shape");
    near(sum_all(a).data(), {21}, "sum all");
    near(max(a, 1).data(), {3, 6}, "max");
    near(amax(a, 0).data(), {4, 5, 6}, "amax");
    near(min(a, 1).data(), {1, 4}, "min");
    near(amin(a, 0).data(), {1, 2, 3}, "amin");
    near(prod(a, 1).data(), {6, 120}, "prod");
    near(transpose(a, 0, 1).data(), {1, 4, 2, 5, 3, 6}, "transpose");
    near(reshape(a, {3, -1}).data(), a.data(), "reshape");
    near(slice(a, 1, 0, 3, 2).data(), {1, 3, 4, 6}, "slice");
    near(concat({a, a}, 1).data(), {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}, "concat");
    std::vector<Tensor> parts = split(a, 2, 1);
    check(parts.size() == 2, "split count");
    near(parts[1].data(), {3, 6}, "split remainder");
    near(repeat(t({2}, {1, 2}), {2, 2}).data(), {1, 2, 1, 2, 1, 2, 1, 2}, "repeat");
    near(expand(t({2, 1}, {1, 2}), {2, 3}).data(), {1, 1, 1, 2, 2, 2}, "expand");
    check(squeeze(unsqueeze(a, 1)).shape() == a.shape(), "squeeze unsqueeze");
    Tensor ix = ids({2, 2}, {2, 0, 1, 1});
    near(gather(a, 1, ix).data(), {3, 1, 5, 5}, "gather");
    near(index_select(a, 0, ids({3}, {1, 0, 1})).data(), {4, 5, 6, 1, 2, 3, 4, 5, 6}, "index select");
    near(scatter(a, 1, ix, t({2, 2}, {9, 8, 7, 6})).data(), {8, 2, 9, 4, 6, 6}, "scatter last wins");
    gradient("gather", {2, 3}, {1, 2, 3, 4, 5, 6}, [ix](const Tensor& x) { return gather(x, 1, ix); });
    gradient("index select duplicates", {2, 2}, {1, 2, 3, 4},
             [](const Tensor& x) { return index_select(x, 0, ids({3}, {1, 0, 1})); });
    gradient("repeat", {2}, {1, 2}, [](const Tensor& x) { return repeat(x, {3, 2}); });
    gradient("slice permute", {2, 3}, {1, 2, 3, 4, 5, 6},
             [](const Tensor& x) { return slice(transpose(x, 0, 1), 0, 0, 3, 2); });
    gradient("scatter source unique indices", {2, 2}, {1, 2, 3, 4},
             [a](const Tensor& x) { return scatter(a, 1, ids({2, 2}, {2, 0, 1, 0}), x); });
    gradient("prod with zero", {2, 3}, {0, 2, 3, 0, 0, 6}, [](const Tensor& x) { return prod(x, 1); });
    Tensor tie = t({3}, {2, 2, 1}, true);
    max(tie, 0).backward();
    near(tie.grad(), {1, 0, 0}, "max first tie gradient");
    tie.zero_grad();
    amax(tie, 0).backward();
    near(tie.grad(), {.5f, .5f, 0}, "amax shared tie gradient");
    expect_throw([a] { reshape(a, {4, 2}); }, "bad reshape");
    expect_throw([a] { index_select(a, 0, ids({1}, {2})); }, "bad index");
}
void linear_and_neural() {
    Tensor a = t({2, 3}, {1, 2, 3, 4, 5, 6}), b = t({3, 2}, {1, 2, 3, 4, 5, 6});
    near(matmul(a, b).data(), {22, 28, 49, 64}, "matmul");
    near(matmul(t({3}, {1, 2, 3}), t({3}, {4, 5, 6})).data(), {32}, "dot");
    near(addmm(t({2}, {1, 2}), a, b, .5f, 2).data(), {44.5f, 57, 98.5f, 129}, "addmm");
    Tensor ba = reshape(a, {1, 2, 3}), bb = repeat(reshape(b, {1, 3, 2}), {2, 1, 1});
    near(matmul(ba, bb).data(), {22, 28, 49, 64, 22, 28, 49, 64}, "broadcast batch matmul");
    near(bmm(repeat(ba, {2, 1, 1}), bb).data(), {22, 28, 49, 64, 22, 28, 49, 64}, "bmm");
    near(triu(a).data(), {1, 2, 3, 0, 5, 6}, "triu");
    near(tril(a, -1).data(), {0, 0, 0, 4, 0, 0}, "tril");
    near(sum(softmax(a, 1), 1).data(), {1, 1}, "softmax normalization");
    near(exp(log_softmax(a)).data(), softmax(a).data(), "log softmax");
    gradient("matmul broadcast", {1, 2, 3}, {.1f, .2f, .3f, .4f, .5f, .6f},
             [bb](const Tensor& x) { return matmul(x, bb); });
    gradient("softmax", {2, 3}, {.1f, .4f, -.5f, .8f, -.2f, .3f}, [a](const Tensor& x) { return mul(softmax(x), a); });
    gradient("log softmax", {2, 3}, {.1f, .4f, -.5f, .8f, -.2f, .3f},
             [a](const Tensor& x) { return mul(log_softmax(x), a); });
    gradient("layer norm", {2, 3}, {.1f, .4f, -.5f, .8f, -.2f, .3f},
             [](const Tensor& x) { return layer_norm(x, t({3}, {1, 2, 3}), t({3}, {0, 0, 0})); });
    gradient("rms norm", {2, 3}, {.1f, .4f, -.5f, .8f, -.2f, .3f},
             [](const Tensor& x) { return rms_norm(x, t({3}, {1, 2, 3})); });
    gradient("batch norm", {2, 2, 1, 2}, {.1f, .4f, -.5f, .8f, -.2f, .3f, .6f, .9f},
             [](const Tensor& x) { return mul(batch_norm(x, t({2}, {1, 2}), t({2}, {0, 0})), x); });
    gradient("group norm", {1, 2, 1, 3}, {.1f, .4f, -.5f, .8f, -.2f, .3f},
             [](const Tensor& x) { return mul(group_norm(x, 1, t({2}, {1, 2}), t({2}, {0, 0})), x); });
    gradient("cross entropy", {2, 3}, {.1f, .4f, -.5f, .8f, -.2f, .3f},
             [](const Tensor& x) { return cross_entropy(x, ids({2}, {1, 0})); });
    gradient("mse", {2}, {.1f, .4f}, [](const Tensor& x) { return mse_loss(x, t({2}, {.3f, -.2f})); });
}
void convolution_and_random() {
    Tensor w = t({1, 1, 2, 2}, {1, 2, 3, 4}), bias = t({1}, {.1f});
    near(conv2d(t({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}), w, bias).data(), {37.1f, 47.1f, 67.1f, 77.1f}, "conv2d");
    gradient(
        "conv input", {1, 1, 3, 3}, {.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f},
        [w, bias](const Tensor& x) { return conv2d(x, w, bias, 2, 1); }, .02f);
    Tensor x = t({1, 1, 3, 3}, {.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f});
    gradient("conv weight", {1, 1, 2, 2}, {.2f, .3f, .4f, .5f},
             [x, bias](const Tensor& v) { return conv2d(x, v, bias, 1, 1); });
    gradient("conv bias", {1}, {.2f}, [x, w](const Tensor& v) { return conv2d(x, w, v); });
    near(max_pool2d(x, 2, 1).data(), {.5f, .6f, .8f, .9f}, "max pool");
    near(avg_pool2d(x, 2, 1).data(), {.3f, .4f, .6f, .7f}, "avg pool");
    gradient("max pool", x.shape(), x.data(), [](const Tensor& v) { return max_pool2d(v, 2, 1); });
    gradient("avg pool", x.shape(), x.data(), [](const Tensor& v) { return avg_pool2d(v, 2, 1, 1); });
    std::vector<float> uniform = Tensor::uniform({64}, -2, 3, 123, device).data();
    near(uniform, Tensor::uniform({64}, -2, 3, 123, device).data(), "uniform deterministic");
    check(std::all_of(uniform.begin(), uniform.end(), [](float v) { return v >= -2 && v <= 3; }), "uniform range");
    std::vector<float> normal = Tensor::normal({64}, 0, 1, 456, device).data();
    near(normal, Tensor::normal({64}, 0, 1, 456, device).data(), "normal deterministic");
    check(std::all_of(normal.begin(), normal.end(), [](float v) { return std::isfinite(v); }), "normal finite");
    Tensor ones = Tensor::ones({64}, device, true);
    Tensor dropped = dropout(ones, .5f, 19);
    sum_all(dropped).backward();
    near(ones.grad(), dropped.data(), "dropout backward mask");
    near(dropout(x, .5f, 19, false).data(), x.data(), "dropout eval");
}
void full_vocabulary_loss() {
    if (device != Backend::CUDA) return;
    constexpr int classes = 50257;
    std::vector<float> values(2 * classes), expected(2 * classes);
    double expected_loss = 0;
    for (int row = 0; row < 2; ++row) {
        double denominator = 0;
        for (int col = 0; col < classes; ++col) {
            values[row * classes + col] = float((col * 17 + row) % 101 - 50) * .05f;
            denominator += std::exp(double(values[row * classes + col]));
        }
        int target = row == 0 ? 0 : classes - 1;
        expected_loss += (std::log(denominator) - values[row * classes + target]) / 2;
        for (int col = 0; col < classes; ++col)
            expected[row * classes + col] = float((std::exp(double(values[row * classes + col])) / denominator - double(col == target)) / 2);
    }
    Tensor input = t({2, classes}, values, true);
    Tensor loss = cross_entropy(input, ids({2}, {0, float(classes - 1)}));
    near(loss.data(), {float(expected_loss)}, "full vocabulary cross entropy", 2e-6f);
    loss.backward();
    near(input.grad(), expected, "full vocabulary loss gradient", 2e-6f);
}
void graph_and_parity() {
    Tensor x = t({2}, {2, 3}, true);
    Tensor y = sum_all(mul(x, x));
    y.backward();
    near(x.grad(), {4, 6}, "shared parent gradient");
    y.backward();
    near(x.grad(), {8, 12}, "leaf accumulation");
    x.zero_grad();
    check(x.grad().empty(), "zero grad");
    check(y.graph_json().find("elman.graph") != std::string::npos, "graph export");
    x.assign({3, 4});
    expect_throw([y]() mutable { y.backward(); }, "mutation version guard");
    Tensor original = t({2}, {1, 2}, true);
    Tensor result = sum_all(exp(original));
    original.detach().assign({3, 4});
    expect_throw([result]() mutable { result.backward(); }, "detached alias mutation guard");
    Tensor output = exp(t({2}, {1, 2}, true));
    output.detach().assign({3, 4});
    expect_throw([output]() mutable { sum_all(output).backward(); }, "saved output mutation guard");
    if (device != Backend::CPU) {
        Tensor a = Tensor::normal({2, 3, 4}, 0, .2f, 23, device);
        Tensor b = Tensor::normal({1, 4, 3}, 0, .3f, 47, device);
        Tensor (*const run)(Tensor, Tensor) = [](Tensor a, Tensor b) { return sum(softmax(matmul(a, b)), 1); };
        near(run(a, b).data(), run(a.to(Backend::CPU), b.to(Backend::CPU)).data(), "CPU backend parity", .0002f);
        near(Tensor::uniform({64}, 0, 1, 42, device).data(), Tensor::uniform({64}, 0, 1, 42, Backend::CPU).data(),
             "random backend parity");
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        device = argc > 1 ? parse_backend(argv[1]) : Backend::CPU;
        check(parse_backend("cpu") == Backend::CPU && parse_backend("cuda") == Backend::CUDA &&
                  parse_backend("metal") == Backend::Metal && backend_name(Backend::Metal) == "metal",
              "registered backend names");
        expect_throw([] { parse_backend("compute"); }, "removed Compute backend");
        expect_throw([] { parse_backend("d3d11"); }, "removed D3D11 alias");
#ifndef ELMAN_HAS_METAL
        check(!backend_available(Backend::Metal), "Metal availability without a compiled backend");
#endif
        if (!backend_available(device)) {
            std::cerr << "Backend unavailable: " << backend_name(device) << '\n';
            return 77;
        }
        arithmetic();
        reductions_and_layout();
        linear_and_neural();
        convolution_and_random();
        graph_and_parity();
        full_vocabulary_loss();
        std::cout << "PASS backend=" << backend_name(device) << " checks=" << checks
                  << " native_dispatches=" << backend_dispatch_count(device) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
