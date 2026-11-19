#pragma once
#include "Backend.h"
#include <functional>
#include <complex>
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
    static Tensor from_int_data(Shape shape, const std::vector<int32_t>& data, Backend backend = Backend::CPU,
                                DType dtype = DType::Int32);
    static Tensor from_double_data(Shape shape, const std::vector<double>& data, Backend backend = Backend::CPU, bool requires_grad = false);
    static Tensor from_complex_data(Shape shape, const std::vector<std::complex<double>>& data,
                                    Backend backend = Backend::CPU, bool requires_grad = false, DType dtype = DType::Complex64);
    static Tensor from_int64_data(Shape shape, const std::vector<int64_t>& data, Backend backend = Backend::CPU);
    static Tensor from_raw(Shape shape, DType dtype, const std::vector<uint8_t>& bytes,
                           Backend backend = Backend::CPU, bool requires_grad = false);
    static Tensor zeros(Shape shape, Backend backend = Backend::CPU, bool requires_grad = false, DType dtype = DType::Float32);
    static Tensor ones(Shape shape, Backend backend = Backend::CPU, bool requires_grad = false, DType dtype = DType::Float32);
    static Tensor uniform(Shape shape, double low, double high, uint32_t seed, Backend backend = Backend::CPU,
                          bool requires_grad = false, DType dtype = DType::Float32);
    static Tensor normal(Shape shape, double mean, double stddev, uint32_t seed, Backend backend = Backend::CPU,
                         bool requires_grad = false, DType dtype = DType::Float32);
    const Shape& shape() const;
    size_t numel() const;
    size_t nbytes() const;
    Backend backend() const;
    int device() const;
    DType dtype() const;
    bool defined() const { return bool(node_); }
    bool requires_grad() const;
    std::vector<float> data() const;
    std::vector<double> double_data() const;
    std::vector<std::complex<double>> complex_data() const;
    std::vector<std::complex<double>> complex_grad() const;
    void assign_complex(const std::vector<std::complex<double>>& values);
    std::vector<double> double_grad() const;
    void assign_double(const std::vector<double>& values);
    std::vector<int32_t> int_data() const;
    std::vector<int64_t> int64_data() const;
    std::vector<uint8_t> raw_data() const;
    std::vector<float> grad() const;
    void backward();
    void backward(const Tensor& gradient);
    void zero_grad();
    void assign(const std::vector<float>& values);
    void assign_int(const std::vector<int32_t>& values);
    void assign_int64(const std::vector<int64_t>& values);
    void assign_raw(const std::vector<uint8_t>& bytes);
    // Inference-only indexed mutation; overlapping source/index storage is rejected.
    void assign_indexed(int dim, const Tensor& indices, const Tensor& source);
    Tensor detach() const;
    Tensor to(Backend backend) const;
    Tensor to(Backend backend, int device) const;
    Tensor to_device(int device) const;
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
                        Tensor add(const Tensor& a, double b);
Tensor sub(const Tensor& a, double b);
Tensor mul(const Tensor& a, double b);
Tensor div(const Tensor& a, double b);
Tensor pow(const Tensor& a, double b);
Tensor leaky_relu(const Tensor& x, double slope = 0.01);
Tensor where(const Tensor& condition, const Tensor& a, const Tensor& b);
Tensor cast(const Tensor& x, DType dtype);
Tensor complex(const Tensor& real, const Tensor& imaginary);
Tensor real(const Tensor& x);
Tensor imag(const Tensor& x);
Tensor conj(const Tensor& x);
Tensor angle(const Tensor& x);
Tensor sgn(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor reshape(const Tensor& x, Shape shape);
// Zero-copy contiguous subrange. Offset is in elements; mutation/version and
// allocation lifetime are shared with the source. Shape has explicit dimensions.
Tensor contiguous_view(const Tensor& x, Shape shape, size_t offset = 0);
Tensor permute(const Tensor& x, const std::vector<int>& axes);
Tensor transpose(const Tensor& x, int a, int b);
Tensor slice(const Tensor& x, int dim, int start, int stop, int step = 1);
Tensor gather(const Tensor& x, int dim, const Tensor& indices);
Tensor scatter(const Tensor& x, int dim, const Tensor& indices, const Tensor& source);
Tensor index_select(const Tensor& x, int dim, const Tensor& indices);
// Deterministic index_copy: duplicate indices use the final source entry.
// Indices support Int64 and the framework's Int32 indexing extension.
Tensor index_copy(const Tensor& x, int dim, const Tensor& indices, const Tensor& source);
// Stable ordering; NaNs sort last ascending and first descending. Indices use
// the framework's Int32 indexing ABI and retain input order for equal values.
Tensor argsort(const Tensor& x, int dim = -1, bool descending = false);
std::pair<Tensor, Tensor> sort(const Tensor& x, int dim = -1, bool descending = false);
Tensor concat(const std::vector<Tensor>& inputs, int dim);
std::vector<Tensor> split(const Tensor& x, int size, int dim = 0);
Tensor repeat(const Tensor& x, const Shape& repeats);
Tensor expand(const Tensor& x, const Shape& shape);
Tensor squeeze(const Tensor& x, int dim);
Tensor squeeze(const Tensor& x);
Tensor unsqueeze(const Tensor& x, int dim);
#define ELMAN_REDUCE(name) Tensor name(const Tensor& x, int dim = -1, bool keepdim = false);
// Sum and product promote Boolean and byte-addressable integers to Int64.
ELMAN_REDUCE(sum)
ELMAN_REDUCE(mean) ELMAN_REDUCE(max) ELMAN_REDUCE(min) ELMAN_REDUCE(amax) ELMAN_REDUCE(amin) ELMAN_REDUCE(prod)
#undef ELMAN_REDUCE
    Tensor sum_all(const Tensor& x);
Tensor mean_all(const Tensor& x);
// Inclusive prefix sum. Integer/Bool inputs promote to Int64 by default.
Tensor cumsum(const Tensor& x, int dim = -1);
Tensor dropout(const Tensor& x, double probability, uint32_t seed, bool training = true);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_transposed(const Tensor& a, const Tensor& b);
Tensor bmm(const Tensor& a, const Tensor& b);
Tensor addmm(const Tensor& c, const Tensor& a, const Tensor& b, double beta = 1, double alpha = 1);
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias = Tensor{});
// Weight/bias concatenate output features. Output concatenates each projection's
// flattened [..., output_sizes[i]] result, preserving separate GEMM arithmetic.
Tensor packed_linear(const Tensor& input, const Tensor& weight, const std::vector<int>& output_sizes,
                     const Tensor& bias = Tensor{});
Tensor triu(const Tensor& x, int diagonal = 0);
Tensor tril(const Tensor& x, int diagonal = 0);
Tensor softmax(const Tensor& x, int dim = -1);
// Match cast(softmax(cast(scores / divisor + mask, Float32), dim), scores.dtype()).
// Division and mask addition round separately in the shared input dtype.
Tensor attention_softmax(const Tensor& scores, const Tensor& mask, double divisor = 1, int dim = -1);
// Q [B,Hq,Q,D], K [B,Hkv,K,D], V [B,Hkv,K,Dv]; additive mask broadcasts
// to [B,Hq,Q,K]. Hq must be divisible by Hkv. Forward bounds score scratch
// by query_tile rows, or by one complete Q*K head in Heads mode, and shares
// K/V heads. Causal masking replaces masked values
// above key=query+causal_offset with the dtype minimum, as in eager Qwen2.
// Backward recomputes eager attention.
inline constexpr int kAttentionQueryTile = 32;
enum class AttentionTiling { Queries = 0, Heads = 1 };
Tensor tiled_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                       const Tensor& mask, double divisor = 1, int query_tile = kAttentionQueryTile,
                       bool causal = false, int causal_offset = 0,
                       AttentionTiling tiling = AttentionTiling::Queries);
Tensor log_softmax(const Tensor& x, int dim = -1);
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, double eps = 1e-5);
// Output rounds the final result. Normalized additionally rounds
// the normalized activation to the input dtype before applying the weight.
// Output permits mixed floating weights; Normalized requires matching dtypes.
enum class RmsNormRounding { Output, Normalized };
Tensor rms_norm(const Tensor& x, const Tensor& weight, double eps = 1e-5,
                RmsNormRounding rounding = RmsNormRounding::Output);
// Preserve the intermediate dtype rounding of silu(gate) * up and split-half
// rotary embeddings. Operands have one floating dtype; broadcasting is allowed.
Tensor silu_mul(const Tensor& gate, const Tensor& up);
Tensor rotary_embedding(const Tensor& x, const Tensor& cosine, const Tensor& sine);
Tensor batch_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, double eps = 1e-5);
Tensor group_norm(const Tensor& x, int groups, const Tensor& weight, const Tensor& bias, double eps = 1e-5);
Tensor mse_loss(const Tensor& prediction, const Tensor& target);
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride = 1, int padding = 0);
Tensor max_pool2d(const Tensor& x, int kernel, int stride = 0, int padding = 0);
Tensor avg_pool2d(const Tensor& x, int kernel, int stride = 0, int padding = 0);
Tensor upsample_nearest2d(const Tensor& x, int scale_height = 2, int scale_width = 2);
} // namespace elman
