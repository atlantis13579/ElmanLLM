#include "Operators/inference_ops.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "Core/TensorDispatch.h"
#include "Operators/gelu_params.h"

namespace elman {

Tensor exact_gelu(const Tensor& value) {
  if (!is_compute_float(value.dtype())) throw std::invalid_argument("Exact GELU requires a compute floating dtype");
  const Tensor input = cast(value, DType::Float32);
  std::vector<int> attributes(operator_attr::gelu::kExact + 1, 0);
  attributes[operator_attr::gelu::kExact] = 1;
  return cast(dispatch_tensor("gelu", input.shape(), {input}, attributes, input.dtype()), value.dtype());
}

Tensor quick_gelu(const Tensor& value) {
  if (!is_compute_float(value.dtype())) throw std::invalid_argument("Quick GELU requires a compute floating dtype");
  Tensor scaled = mul(cast(value, DType::Float32), 1.702);
  if (is_narrow_float(value.dtype())) scaled = cast(cast(scaled, value.dtype()), DType::Float32);
  Tensor one = Tensor::ones({}, value.backend());
  Tensor sigmoid = div(one, add(exp(mul(scaled, -1.0)), 1.0));
  return mul(value, cast(sigmoid, value.dtype()));
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value, bool causal) {
  if (query.shape().size() != 4 || query.shape().back() <= 0 ||
      query.dtype() != key.dtype() || query.dtype() != value.dtype())
    throw std::invalid_argument("Attention requires a valid head width and matching dtypes");
  const DType dtype = query.dtype();
  const DType accumulation = is_narrow_float(dtype) ? DType::Float32 : dtype;
  Tensor zero = Tensor::zeros({}, query.backend(), false, accumulation);
  Tensor output = tiled_attention(cast(query, accumulation), cast(key, accumulation), cast(value, accumulation),
                                zero, std::sqrt(query.shape().back()), kAttentionQueryTile, causal);
  return cast(output, dtype);
}

Tensor conv2d_with_channel_bias(const Tensor& input, const Tensor& weight, const Tensor& bias, int stride, int padding) {
  if (bias.defined() && input.backend() == Backend::CUDA && is_narrow_float(input.dtype())) {
    if (weight.shape().empty() || bias.shape() != Shape{weight.shape().front()} || bias.dtype() != input.dtype())
      throw std::invalid_argument("Convolution bias must match the output channels and input dtype");
    // Round the convolution result before adding the channel bias.
    Tensor result = conv2d(input, weight, {}, stride, padding);
    return add(result, reshape(bias, {1, bias.shape()[0], 1, 1}));
  }
  return conv2d(input, weight, bias, stride, padding);
}

Tensor image_tokens(const Tensor& image) {
  const Shape& shape = image.shape();
  if (shape.size() != 4 || int64_t(shape[2]) * shape[3] > std::numeric_limits<int>::max())
    throw std::invalid_argument("Image tokenization requires NCHW input with a valid spatial extent");
  return transpose(reshape(image, {shape[0], shape[1], shape[2] * shape[3]}), 1, 2);
}

Tensor token_image(const Tensor& tokens, int height, int width) {
  const Shape& shape = tokens.shape();
  if (shape.size() != 3 || height < 0 || width < 0 || int64_t(height) * width != shape[1])
    throw std::invalid_argument("Token count must match the image extent");
  return reshape(transpose(tokens, 1, 2), {shape[0], shape[2], height, width});
}

Tensor sinusoidal_embedding(int step, int width, double frequency_shift, bool cosine_first, Backend backend, DType dtype) {
  const int half = width / 2;
  if (step < 0 || width <= 0 || !std::isfinite(frequency_shift) || half <= frequency_shift)
    throw std::invalid_argument("Invalid sinusoidal embedding dimensions or timestep");
  std::vector<float> values(width, 0);
  for (int frequency = 0; frequency < half; ++frequency) {
    const double phase = step * std::exp(-std::log(10000.0) * frequency / (half - frequency_shift));
    values[frequency + (cosine_first ? half : 0)] = static_cast<float>(std::sin(phase));
    values[frequency + (cosine_first ? 0 : half)] = static_cast<float>(std::cos(phase));
  }
  return Tensor::from_data({1, width}, values, backend, false, dtype);
}

}  // namespace elman
