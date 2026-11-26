#pragma once

#include "Core/Tensor.h"

namespace elman {

Tensor exact_gelu(const Tensor& value);
Tensor quick_gelu(const Tensor& value);
Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value, bool causal);
Tensor conv2d_with_channel_bias(const Tensor& input, const Tensor& weight, const Tensor& bias, int stride, int padding);
Tensor image_tokens(const Tensor& image);
Tensor token_image(const Tensor& tokens, int height, int width);
Tensor sinusoidal_embedding(int step, int width, double frequency_shift, bool cosine_first, Backend backend, DType dtype);

}  // namespace elman
