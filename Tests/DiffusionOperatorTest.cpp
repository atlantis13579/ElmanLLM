#include <cmath>
#include <iostream>
#include <stdexcept>

#include "Core/TensorDispatch.h"
#include "Kernals/flux2_ops.h"

namespace {
using namespace elman;
using namespace elman::flux2;
Backend backend;

Tensor data(Shape shape, uint32_t salt, float scale = 1.f, float offset = 0.f, DType dtype = DType::Float32,
            Backend location = backend) {
  std::vector<float> values(shape_element_count(shape));
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t x = uint32_t(i) + salt * 100003u;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x ^= x >> 16;
    values[i] = (int(x % 257) - 128) / 128.f * scale + offset;
  }
  return Tensor::from_data(shape, values, location, false, dtype);
}

void close(float actual, float expected, float tolerance, const char* operation) {
  if (!(std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)))) {
    throw std::runtime_error(std::string(operation) + ": " + std::to_string(actual) + " vs " +
                             std::to_string(expected));
  }
}
void compare(const Tensor& actual, const Tensor& expected, float tolerance, const char* operation) {
  if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype() || actual.backend() != backend) {
    throw std::runtime_error(std::string(operation) + " descriptor mismatch");
  }
  std::vector<float> a = actual.data(), b = expected.data();
  for (size_t i = 0; i < a.size(); ++i) {
    close(a[i], b[i], tolerance, operation);
  }
}

void primitives(DType type) {
  const float tolerance = type == DType::Float32 ? 3e-5f : type == DType::BFloat16 ? .006f : .001f;
  Tensor ids = Tensor::from_int_data({1, 5, 4},
      {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 1, 0}, backend);
  std::vector<int32_t> coords = ids.int_data();
  Tensor all_ids = concat({ids, add(ids, Tensor::from_int_data({}, {1}, backend))}, 0);
  Tensor x = data({2, 3, 5, 128}, 6, 1, 0, type);
  std::vector<float> y = apply_rotary(x, rotary_table(all_ids)).data(), input = x.data();
  for (size_t i = 0; i < y.size(); ++i) {
    const int pair = (i % 32) / 2, axis = (i % 128) / 32, token = (i / 128) % 5;
    const double angle = (coords[token * 4 + axis] + int(i / (3 * 5 * 128))) * std::pow(2000., -pair / 16.);
    close(y[i], float(input[i] * std::cos(angle) + (i % 2 ? 1 : -1) * input[i ^ 1] * std::sin(angle)), tolerance,
          "RoPE");
  }

  Tensor moments = data({2, 64, 4, 6}, 2, 1, 0, type);
  Tensor mean = data({128}, 3, .25f), var = data({128}, 4, .25f, .5f);
  Tensor packed = pack_latents(moments, mean, var);
  if (packed.shape() != Shape{2, 128, 2, 3}) {
    throw std::runtime_error("Invalid packed latent shape");
  }
  std::vector<float> values = moments.data(), m = mean.data(), v = var.data(), z = packed.data();
  for (int b = 0; b < 2; ++b) {
    for (int c = 0; c < 128; ++c) {
      for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 3; ++w) {
          const int i = ((b * 64 + c / 4) * 4 + 2 * h + (c / 2) % 2) * 6 + 2 * w + c % 2;
          close(z[((b * 128 + c) * 2 + h) * 3 + w], (values[i] - m[c]) / std::sqrt(v[c] + 1e-4f), tolerance, "pack");
        }
      }
    }
  }
  compare(unpack_latents(packed, mean, var), slice(moments, 1, 0, 32), tolerance, "unpack");

  // Both sequence lengths end in partial GPU tiles. Compare to independently
  // materialized FP32 logits so the CPU kernel is also checked against SDPA.
  Tensor q = data({2, 2, 17, 128}, 4, 2, 0, type), k = data({2, 2, 37, 128}, 5, 2, 0, type);
  Tensor value = data(k.shape(), 6, 1, 0, type);
  Tensor scores = div(matmul_transposed(cast(q, DType::Float32), cast(k, DType::Float32)), std::sqrt(128.));
  Tensor expected = cast(matmul(softmax(scores, -1), cast(value, DType::Float32)), type);
  compare(attention(q, k, value), expected, tolerance, "SDPA");

  Tensor features = data({2, 3, 131}, 72, 2, 0, type), shift = data({2, 1, 131}, 73, 1, 0, type),
       scale = data(shift.shape(), 74, 1, 0, type);
  Tensor work = cast(features, DType::Float32);
  Tensor centered = sub(work, elman::mean(work, -1, true));
  Tensor norm = cast(mul(centered, rsqrt(add(elman::mean(mul(centered, centered), -1, true), 1e-6))), type);
  Tensor mod = add(mul(norm, add(scale, 1.0)), shift);
  compare(modulate(features, shift, scale), mod, tolerance, "adaptive LayerNorm");
}

void timestep_reference() {
  const auto values = timestep_embedding(Tensor::from_data({2}, {.125f, 4.f}, backend)).data();
  for (int b = 0; b < 2; ++b) {
    for (int j = 0; j < 256; ++j) {
      // The operator stores frequencies and computes phases in FP32.
      const float phase = (b == 0 ? 125.f : 4000.f) * float(std::pow(10000., -double(j % 128) / 128));
      close(values[b * 256 + j], float(j < 128 ? std::cos(phase) : std::sin(phase)), 1e-5f, "time");
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("Expected a backend argument");
    backend = parse_backend(argv[1]);
    if (!backend_available(backend)) return 77;
    for (DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) primitives(dtype);
    timestep_reference();
    std::cout << "Diffusion operator checks passed on " << backend_name(backend) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
