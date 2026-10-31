#pragma once
#include "backend.h"
#include <functional>
#include <initializer_list>
#include <utility>

namespace elman {
using Shape = std::vector<int>;
struct TensorNode;
class Tensor {
  public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorNode> node) : node_(std::move(node)) {}
    static Tensor from_data(Shape shape, const std::vector<float>& data, Backend backend = Backend::CPU,
                            bool requires_grad = false, DType dtype = DType::Float32);
    static Tensor zeros(Shape shape, Backend backend = Backend::CPU, bool requires_grad = false);
    static Tensor ones(Shape shape, Backend backend = Backend::CPU, bool requires_grad = false);
    static Tensor uniform(Shape shape, float low, float high, uint32_t seed, Backend backend = Backend::CPU,
                          bool requires_grad = false);
    static Tensor normal(Shape shape, float mean, float stddev, uint32_t seed, Backend backend = Backend::CPU,
                         bool requires_grad = false);
    const Shape& shape() const;
    size_t numel() const;
    Backend backend() const;
    DType dtype() const;
    bool defined() const { return bool(node_); }
    bool requires_grad() const;
    std::vector<float> data() const;
    std::vector<float> grad() const;
    void backward();
    void backward(const Tensor& gradient);
    void zero_grad();
    void assign(const std::vector<float>& values);
    Tensor detach() const;
    Tensor to(Backend backend) const;
    std::string graph_json() const;
    const std::shared_ptr<TensorNode>& node() const { return node_; }

  private:
    std::shared_ptr<TensorNode> node_;
};

#define ELMAN_BINARY(name) Tensor name(const Tensor& a, const Tensor& b);
ELMAN_BINARY(add)
ELMAN_BINARY(sub) ELMAN_BINARY(mul) ELMAN_BINARY(div) ELMAN_BINARY(pow) ELMAN_BINARY(maximum) ELMAN_BINARY(minimum)
    ELMAN_BINARY(eq) ELMAN_BINARY(ne) ELMAN_BINARY(gt) ELMAN_BINARY(ge) ELMAN_BINARY(lt) ELMAN_BINARY(le)
        ELMAN_BINARY(logical_and) ELMAN_BINARY(logical_or)
#undef ELMAN_BINARY
#define ELMAN_UNARY(name) Tensor name(const Tensor& x);
            ELMAN_UNARY(sqrt) ELMAN_UNARY(rsqrt) ELMAN_UNARY(exp) ELMAN_UNARY(log) ELMAN_UNARY(sin) ELMAN_UNARY(cos)
                ELMAN_UNARY(abs) ELMAN_UNARY(sign) ELMAN_UNARY(logical_not) ELMAN_UNARY(relu) ELMAN_UNARY(silu)
                    ELMAN_UNARY(swish) ELMAN_UNARY(gelu)
#undef ELMAN_UNARY
                        Tensor add(const Tensor& a, float b);
Tensor sub(const Tensor& a, float b);
Tensor mul(const Tensor& a, float b);
Tensor div(const Tensor& a, float b);
Tensor pow(const Tensor& a, float b);
Tensor leaky_relu(const Tensor& x, float slope = 0.01f);
Tensor cast(const Tensor& x, DType dtype);
Tensor reshape(const Tensor& x, Shape shape);
Tensor permute(const Tensor& x, const std::vector<int>& axes);
Tensor transpose(const Tensor& x, int a, int b);
Tensor slice(const Tensor& x, int dim, int start, int stop, int step = 1);
Tensor gather(const Tensor& x, int dim, const Tensor& indices);
Tensor scatter(const Tensor& x, int dim, const Tensor& indices, const Tensor& source);
Tensor index_select(const Tensor& x, int dim, const Tensor& indices);
Tensor concat(const std::vector<Tensor>& inputs, int dim);
std::vector<Tensor> split(const Tensor& x, int size, int dim = 0);
Tensor repeat(const Tensor& x, const Shape& repeats);
Tensor expand(const Tensor& x, const Shape& shape);
Tensor squeeze(const Tensor& x, int dim);
Tensor squeeze(const Tensor& x);
Tensor unsqueeze(const Tensor& x, int dim);
#define ELMAN_REDUCE(name) Tensor name(const Tensor& x, int dim = -1, bool keepdim = false);
ELMAN_REDUCE(sum)
ELMAN_REDUCE(mean) ELMAN_REDUCE(max) ELMAN_REDUCE(min) ELMAN_REDUCE(amax) ELMAN_REDUCE(amin) ELMAN_REDUCE(prod)
#undef ELMAN_REDUCE
    Tensor sum_all(const Tensor& x);
Tensor mean_all(const Tensor& x);
Tensor dropout(const Tensor& x, float probability, uint32_t seed, bool training = true);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor bmm(const Tensor& a, const Tensor& b);
Tensor addmm(const Tensor& c, const Tensor& a, const Tensor& b, float beta = 1, float alpha = 1);
Tensor triu(const Tensor& x, int diagonal = 0);
Tensor tril(const Tensor& x, int diagonal = 0);
Tensor softmax(const Tensor& x, int dim = -1);
Tensor log_softmax(const Tensor& x, int dim = -1);
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps = 1e-5f);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-5f);
Tensor batch_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps = 1e-5f);
Tensor group_norm(const Tensor& x, int groups, const Tensor& weight, const Tensor& bias, float eps = 1e-5f);
Tensor mse_loss(const Tensor& prediction, const Tensor& target);
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride = 1, int padding = 0);
Tensor max_pool2d(const Tensor& x, int kernel, int stride = 0, int padding = 0);
Tensor avg_pool2d(const Tensor& x, int kernel, int stride = 0, int padding = 0);
} // namespace elman
