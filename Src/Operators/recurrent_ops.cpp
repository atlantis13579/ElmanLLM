#include "Operators/recurrent_ops.h"

#include "Core/Device.h"
#include "Core/TensorDispatch.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
void same_device(const Tensor& a, const Tensor& b) {
    require(b.defined() && a.backend() == b.backend() && a.device() == b.device(), "Recurrent operands must share a device");
    require(!b.requires_grad(), "Recurrent inference operators do not record gradients");
}
}

Tensor sigmoid(const Tensor& input) {
    require(is_compute_float(input.dtype()), "Sigmoid requires floating input");
    DeviceScope owner(input.backend(), input.device());
    const Tensor denominator = add(exp(mul(cast(input, DType::Float32), -1)), 1);
    return cast(div(Tensor::ones({}, input.backend()), denominator), input.dtype());
}

RecurrentOutput causal_depthwise_conv1d(const Tensor& input, const Tensor& weight, const Tensor& state) {
    require(input.shape().size() == 3 && weight.shape().size() == 2, "Causal convolution expects BTC input and CK weights");
    same_device(input, input); same_device(input, weight);
    const auto& s = input.shape(); const int kernel = weight.shape()[1];
    require(s[0] > 0 && s[1] > 0 && s[2] > 0 && kernel > 0 && kernel <= 64 && weight.shape()[0] == s[2] &&
        is_compute_float(input.dtype()) && weight.dtype() == input.dtype(), "Invalid causal convolution dimensions or dtype");
    DeviceScope owner(input.backend(), input.device());
    Tensor history = state;
    if (history.defined()) {
        same_device(input, history);
        require(history.shape() == Shape{s[0], kernel - 1, s[2]} && history.dtype() == input.dtype(), "Invalid causal convolution state");
    } else history = Tensor::zeros({s[0], kernel - 1, s[2]}, input.backend(), false, input.dtype());
    const Tensor joined = concat({history, input}, 1);
    const Tensor wide = cast(joined, DType::Float32), coefficients = cast(weight, DType::Float32);
    Tensor output;
    for (int k = 0; k < kernel; ++k) {
        Tensor part = mul(slice(wide, 1, k, k + s[1]), reshape(slice(coefficients, 1, k, k + 1), {1, 1, s[2]}));
        output = output.defined() ? add(output, part) : part;
    }
    return {cast(output, input.dtype()), slice(joined, 1, s[1], s[1] + kernel - 1)};
}

RecurrentOutput gated_delta_rule(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& log_decay, const Tensor& beta, const Tensor& initial) {
    require(query.shape().size() == 4 && key.shape() == query.shape() && value.shape().size() == 4,
            "Gated delta requires BTHD queries, keys and values");
    for (const auto* operand : {&query, &key, &value, &log_decay, &beta}) same_device(query, *operand);
    const auto& s = query.shape(); const auto& v = value.shape();
    const int batch = s[0], time = s[1], heads = v[2], width = s[3], columns = v[3];
    require(batch > 0 && time > 0 && s[2] > 0 && heads > 0 && heads % s[2] == 0 && width > 0 && columns > 0 &&
            v[0] == batch && v[1] == time && is_compute_float(query.dtype()) && key.dtype() == query.dtype() &&
            value.dtype() == query.dtype() && is_compute_float(log_decay.dtype()) && is_compute_float(beta.dtype()) &&
            log_decay.shape() == Shape{batch, time, heads} && beta.shape() == log_decay.shape(), "Invalid gated delta dimensions or dtype");
    const int64_t state_count = int64_t(batch) * heads * width * columns;
    const int64_t token_count = int64_t(batch) * time * heads * columns;
    require(state_count + token_count <= std::numeric_limits<int>::max(), "Gated delta output exceeds indexing range");
    DeviceScope owner(query.backend(), query.device());
    Tensor state = initial;
    if (state.defined()) {
        same_device(query, state);
        require(state.shape() == Shape{batch, heads, width, columns} && state.dtype() == DType::Float32, "Invalid gated delta state");
    } else state = Tensor::zeros({batch, heads, width, columns}, query.backend());
    auto normalize = [&](const Tensor& input) {
        Tensor work = cast(input, DType::Float32);
        return div(work, sqrt(add(sum(mul(work, work), 3, true), 1e-6)));
    };
    Tensor q = mul(normalize(query), 1 / std::sqrt(double(width))), k = normalize(key);
    if (heads != s[2]) {
        std::vector<int32_t> ids(heads);
        for (int i = 0; i < heads; ++i) ids[i] = i / (heads / s[2]);
        const auto indices = Tensor::from_int_data({heads}, ids, query.backend());
        q = index_select(q, 2, indices); k = index_select(k, 2, indices);
    }
    const Tensor packed = concat({q, k, cast(value, DType::Float32)}, 3);
    const Tensor gates = concat({unsqueeze(exp(cast(log_decay, DType::Float32)), 3),
                                unsqueeze(cast(beta, DType::Float32), 3)}, 3);
    std::vector<Tensor> outputs;
    // Bound Metal graph size and the duration of one recurrent GPU submission.
    for (int start = 0; start < time; start += 32) {
        const int count = std::min(32, time - start), output_count = batch * count * heads * columns;
        const auto result = dispatch_tensor("gated_delta", {output_count + int(state_count)},
            {slice(packed, 1, start, start + count), slice(gates, 1, start, start + count), state}, {width, columns});
        outputs.push_back(reshape(slice(result, 0, 0, output_count), {batch, count, heads, columns}));
        state = reshape(slice(result, 0, output_count, output_count + int(state_count)), {batch, heads, width, columns});
    }
    return {cast(concat(outputs, 1), query.dtype()), state};
}

} // namespace elman
