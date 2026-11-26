#include "Backends/CpuParallel.h"
#include "kernel.h"

namespace elman {
namespace {
float round_activation(float x, int dtype) {
  if (dtype == int(DType::BFloat16)) {
    return numeric::bfloat_to_float(numeric::float_to_bfloat(x));
  }
  if (dtype == int(DType::Float16)) {
    return numeric::half_to_float(numeric::float_to_half(x));
  }
  return x;
}
}  // namespace

void launch_flux_rotary_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t*,
                            const KernelParams& p) {
  const int heads = p.output_shape(1), tokens = p.output_shape(2);
  parallel_for_cpu(p.output_count() / 128, 16, [&](int64_t first, int64_t last) {
    for (int row = int(first); row < last; ++row) {
      const int batch = p.input_shape(KernelInput::B, 0) == 1 ? 0 : row / tokens / heads;
      const int table = (batch * tokens + row % tokens) * 128;
      for (int d = 0; d < 128; d += 2) {
        const int i = row * 128 + d;
        const float x = READ_A(i), y = READ_A(i + 1), cs = READ_B(table + d), sn = READ_B(table + d + 1);
        WRITE_OUTPUT(i, x * cs - y * sn);
        WRITE_OUTPUT(i + 1, y * cs + x * sn);
      }
    }
  });
}

void launch_flux_modulate_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                              const KernelParams& p) {
  const int width = p.output_shape(2), dtype = p.output_dtype();
  parallel_for_cpu(p.output_count() / width, 4, [&](int64_t first, int64_t last) {
    for (int row = int(first); row < last; ++row) {
      const int start = row * width;
      const int mod = p.input_shape(KernelInput::B, 1) == 1 ? (row / p.output_shape(1)) * width : start;
      float average = 0, variance = 0;
      for (int d = 0; d < width; ++d) {
        average += READ_A(start + d);
      }
      average /= width;
      for (int d = 0; d < width; ++d) {
        const float delta = READ_A(start + d) - average;
        variance += delta * delta;
      }
      const float inverse = 1.f / std::sqrt(variance / width + 1e-6f);
      for (int d = 0; d < width; ++d) {
        const float normalized = round_activation((READ_A(start + d) - average) * inverse, dtype);
        const float scale = round_activation(1.f + READ_C(mod + d), dtype);
        WRITE_OUTPUT(start + d, round_activation(normalized * scale, dtype) + READ_B(mod + d));
      }
    }
  });
}

void launch_flux_attention_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                               const KernelParams& p) {
  const int queries = p.output_shape(2), keys = p.input_shape(KernelInput::B, 2);
  parallel_for_cpu(p.output_count() / 128, 1, [&](int64_t first, int64_t last) {
    std::vector<float> scores(keys);
    for (int row = int(first); row < last; ++row) {
      const int base = row / queries * keys * 128;
      float query[128], result[128]{};
      for (int d = 0; d < 128; ++d) {
        query[d] = READ_A(row * 128 + d);
      }
      float maximum = -std::numeric_limits<float>::infinity();
      for (int key = 0; key < keys; ++key) {
        float dot = 0;
        for (int d = 0; d < 128; ++d) {
          dot += query[d] * READ_B(base + key * 128 + d);
        }
        scores[key] = dot / std::sqrt(128.f);
        maximum = std::max(maximum, scores[key]);
      }
      float sum = 0;
      for (int key = 0; key < keys; ++key) {
        const float weight = std::exp(scores[key] - maximum);
        sum += weight;
        for (int d = 0; d < 128; ++d) {
          result[d] += weight * READ_C(base + key * 128 + d);
        }
      }
      for (int d = 0; d < 128; ++d) {
        WRITE_OUTPUT(row * 128 + d, result[d] / sum);
      }
    }
  });
}
}  // namespace elman
