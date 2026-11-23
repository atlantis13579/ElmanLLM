#include "Operators/low_precision_attention.h"
#include "Core/LocalFunction.h"
#include "Core/TensorDispatch.h"
#include "Core/Graph.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace elman {
namespace {
Tensor wrap(const BufferPtr& buffer, Shape shape) {
    std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>();
    node->shape = std::move(shape); node->value = buffer;
    return Tensor(node);
}
Tensor operand(const std::vector<BufferPtr>& inputs, int i, const KernelParams& p) {
    Shape shape;
    for (int d = 0; d < p.input_rank(KernelInput(i)); ++d) shape.push_back(p.input_shape(KernelInput(i), d));
    return wrap(inputs[i], std::move(shape));
}
double divisor(const KernelParams& p, int slot) {
    if (p.attribute(7)) {
        const uint64_t bits = uint64_t(uint32_t(p.attribute(5))) | (uint64_t(uint32_t(p.attribute(6))) << 32);
        double value; std::memcpy(&value, &bits, sizeof(value)); return value;
    }
    const int bits = p.attribute(slot); float value;
    std::memcpy(&value, &bits, sizeof(value)); return value;
}
Tensor native_softmax(const Tensor& input, int axis, const Tensor& gradient = {}) {
    std::vector<int> attributes(9);
    attributes[0] = axis; attributes[8] = gradient.defined() ? 1 : 2;
    return dispatch_tensor("attention_softmax", input.shape(), gradient.defined() ? std::vector<Tensor>{input, gradient} : std::vector<Tensor>{input}, attributes, DType::Float32);
}
Tensor native_matrix(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false) {
    // Keep transpose flags through vendor dispatch; materializing the transpose
    // can select a different accumulation order for the backward products.
    std::vector<int> attributes(9); attributes[0] = transpose_a; attributes[1] = transpose_b; attributes[8] = 4;
    return dispatch_tensor("attention_softmax", {a.shape()[0], a.shape()[1], a.shape()[transpose_a ? 3 : 2], b.shape()[transpose_b ? 2 : 3]}, {a, b}, attributes, DType::Float32);
}
Tensor expand_heads(const Tensor& input, int heads) {
    const Shape& s = input.shape();
    const int groups = heads / s[1];
    return reshape(expand(reshape(input, {s[0], s[1], 1, s[2], s[3]}), {s[0], s[1], groups, s[2], s[3]}), {s[0], heads, s[2], s[3]});
}
Tensor reduce_heads(const Tensor& gradient, const Shape& shape, int heads) {
    Tensor grouped = reshape(gradient, {shape[0], shape[1], heads / shape[1], shape[2], shape[3]});
    if (heads == shape[1]) return reshape(grouped, shape);
    std::vector<int> attrs(9); attrs[0] = 4; attrs[8] = 5;
    return reshape(dispatch_tensor("attention_softmax", {shape[0], shape[1], 1, shape[2], shape[3]}, {grouped}, attrs, DType::Float32), shape);
}
Tensor reduce_shape(Tensor gradient, const Shape& shape) {
    // Sum all broadcast axes in one FP32 reduction, before leaf quantization.
    Shape target(gradient.shape().size(), 1);
    std::copy(shape.begin(), shape.end(), target.end() - shape.size());
    std::vector<int> attrs(9); attrs[8] = 5;
    for (int d = 0; d < int(target.size()); ++d)
        if (target[d] == 1 && gradient.shape()[d] != 1) attrs[0] |= 1 << d;
    return reshape(attrs[0] ? dispatch_tensor("attention_softmax", target, {gradient}, attrs, DType::Float32) : gradient, shape);
}
Tensor probabilities(const Tensor& query, const Tensor& keys, const Tensor& mask, double scale,
                     bool causal, int64_t offset, Tensor* upper_output = nullptr) {
    Tensor effective = mask, upper;
    const int rows = query.shape()[2], columns = keys.shape()[2];
    if (causal && rows && columns && offset < int64_t(columns) - 1) {
        upper = offset < 1 - int64_t(rows) ? Tensor::ones({}, query.backend(), false, DType::Bool) :
            triu(Tensor::ones({rows, columns}, query.backend(), false, DType::Bool), int(offset) + 1);
        effective = where(upper, Tensor::from_data({}, {-std::numeric_limits<float>::max()}, query.backend()), mask);
    }
    if (upper_output) *upper_output = upper;
    return native_softmax(add(div(native_matrix(query, keys, false, true), scale), effective), 3);
}
Tensor select_mask(Tensor mask, int start, int rows, int batch = -1, int head = -1) {
    const int rank = int(mask.shape().size());
    if (batch >= 0 && rank == 4 && mask.shape()[0] != 1) mask = slice(mask, 0, batch, batch + 1);
    if (head >= 0 && rank >= 3 && mask.shape()[rank - 3] != 1) mask = slice(mask, rank - 3, head, head + 1);
    if (rank >= 2 && mask.shape()[rank - 2] != 1) mask = slice(mask, rank - 2, start, start + rows);
    return mask;
}
Tensor tiled_forward(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& mask, const KernelParams& p) {
    const Shape& q = query.shape(); const Shape& v = value.shape();
    const Shape shape{q[0], q[1], q[2], v[3]};
    if (!shape_element_count(shape)) return Tensor::zeros(shape, query.backend());
    const Tensor keys = expand_heads(key, q[1]), values = expand_heads(value, q[1]);
    const double scale = divisor(p, 0);
    if (!p.attribute(4)) {
        std::vector<Tensor> tiles;
        for (int start = 0; start < q[2];) {
            const int rows = std::min(p.attribute(1), q[2] - start);
            Tensor probability = probabilities(slice(query, 2, start, start + rows), keys, select_mask(mask, start, rows), scale,
                                             p.attribute(2) != 0, int64_t(p.attribute(3)) + start);
            tiles.push_back(native_matrix(probability, values)); start += rows;
        }
        return concat(tiles, 2);
    }
    std::vector<Tensor> heads;
    for (int batch = 0; batch < q[0]; ++batch) for (int head = 0; head < q[1]; ++head) {
        LocalFunction select{[&](const Tensor& input) { return slice(slice(input, 0, batch, batch + 1), 1, head, head + 1); }};
        Tensor probability = probabilities(select(query), select(keys), select_mask(mask, 0, q[2], batch, head), scale,
                                         p.attribute(2) != 0, p.attribute(3));
        heads.push_back(native_matrix(probability, select(values)));
    }
    return reshape(concat(heads, 1), shape);
}
}
void execute_low_precision_attention(LowPrecisionAttention op, const BufferPtr& output,
                                      const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    // Tensor orchestration stays on the selected backend and preserves its
    // stream/event dependencies. Only the final result returns to FP8/FP4.
    const Tensor a = cast(operand(inputs, 0, p), DType::Float32), b = cast(operand(inputs, 1, p), DType::Float32);
    Tensor result;
    if (op == LowPrecisionAttention::Softmax) result = native_softmax(add(div(a, divisor(p, 1)), b), p.attribute(0));
    else result = tiled_forward(a, b, cast(operand(inputs, 2, p), DType::Float32), cast(operand(inputs, 3, p), DType::Float32), p);
    dispatch_tensor_into("copy", wrap(output, result.shape()), {cast(result, output->dtype)});
}
Tensor low_attention_reduce_shape(const Tensor& gradient, const Shape& shape) { return reduce_shape(gradient, shape); }
Tensor low_attention_softmax_derivative(const Tensor& scores, const Tensor& mask, double scale, int axis, const Tensor& gradient) {
    const Tensor input = add(div(cast(scores.detach(), DType::Float32), scale), cast(mask.detach(), DType::Float32));
    return native_softmax(native_softmax(input, axis), axis, cast(gradient, DType::Float32));
}
std::vector<Tensor> low_tiled_attention_derivatives(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& mask, double scale, bool causal, int offset, const Tensor& gradient) {
    const Tensor q = cast(query.detach(), DType::Float32), k = cast(key.detach(), DType::Float32),
               v = cast(value.detach(), DType::Float32), m = cast(mask.detach(), DType::Float32), g = cast(gradient, DType::Float32);
    const int heads = q.shape()[1];
    const Tensor keys = expand_heads(k, heads), values = expand_heads(v, heads);
    Tensor upper;
    const Tensor probability = probabilities(q, keys, m, scale, causal, offset, &upper);
    std::vector<Tensor> result(4);
    if (value.requires_grad()) result[2] = reduce_heads(native_matrix(probability, g, true), value.shape(), heads);
    if (query.requires_grad() || key.requires_grad() || mask.requires_grad()) {
        const Tensor dz = native_softmax(probability, 3, native_matrix(g, values, false, true));
        if (mask.requires_grad()) result[3] = reduce_shape(upper.defined() ? where(upper, Tensor::zeros({}, query.backend()), dz) : dz, mask.shape());
        if (query.requires_grad()) result[0] = native_matrix(div(dz, scale), keys);
        if (key.requires_grad()) result[1] = reduce_heads(native_matrix(div(dz, scale), q, true), key.shape(), heads);
    }
    return result;
}
} // namespace elman
