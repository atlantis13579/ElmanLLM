#include "Operators/flux2_ops.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "Core/Device.h"
#include "Core/TensorDispatch.h"

namespace elman::flux2 {
namespace {
void activation(const Tensor& x, size_t rank) {
  if (x.shape().size() != rank || !is_compute_float(x.dtype())) {
    throw std::invalid_argument("Invalid FLUX.2 activation rank or dtype");
  }
}

Tensor frequencies(int count, double base, Backend backend) {
  std::vector<float> values(count);
  for (int i = 0; i < count; ++i) {
    values[i] = float(std::pow(base, -double(i) / count));
  }
  return Tensor::from_data({count}, values, backend);
}

void check_stats(const Tensor& x, const Tensor& mean, const Tensor& variance) {
  if (mean.shape() != Shape{128} || mean.shape() != variance.shape() || !is_compute_float(mean.dtype()) ||
      !is_compute_float(variance.dtype()) || mean.backend() != x.backend() || variance.backend() != x.backend() ||
      mean.device() != x.device() || variance.device() != x.device()) {
    throw std::invalid_argument("FLUX.2 requires 128 running statistics on the latent device");
  }
}

Tensor channel_stats(const Tensor& values) {
  return reshape(cast(values, DType::Float32), {1, 128, 1, 1});
}
}  // namespace

Tensor rotary_table(const Tensor& ids) {
  const auto& s = ids.shape();
  if (s.size() != 3 || s[2] != 4 ||
      (ids.dtype() != DType::Int32 && ids.dtype() != DType::Int64 && ids.dtype() != DType::Float32)) {
    throw std::invalid_argument("FLUX.2 RoPE needs [B,L,4] coordinates");
  }
  DeviceScope owner(ids.backend(), ids.device());
  auto phases = mul(unsqueeze(cast(ids, DType::Float32), -1), frequencies(16, 2000, ids.backend()));
  return reshape(concat({unsqueeze(cos(phases), -1), unsqueeze(sin(phases), -1)}, -1), {s[0], s[1], 128});
}

Tensor apply_rotary(const Tensor& heads, const Tensor& table) {
  return dispatch_tensor("flux_rotary", heads.shape(), {heads, table}, {}, heads.dtype());
}

Tensor pack_latents(const Tensor& posterior, const Tensor& mean, const Tensor& variance) {
  activation(posterior, 4);
  check_stats(posterior, mean, variance);
  const auto& s = posterior.shape();
  if ((s[1] != 32 && s[1] != 64) || s[2] % 2 || s[3] % 2) {
    throw std::invalid_argument("FLUX.2 requires 32 mean channels and even spatial extents");
  }
  DeviceScope owner(posterior.backend(), posterior.device());
  auto x = s[1] == 32 ? posterior : slice(posterior, 1, 0, 32);
  const int h = s[2] / 2, w = s[3] / 2;
  x = permute(reshape(x, {s[0], 32, h, 2, w, 2}), {0, 1, 3, 5, 2, 4});
  x = cast(reshape(x, {s[0], 128, h, w}), DType::Float32);
  return cast(mul(sub(x, channel_stats(mean)), rsqrt(add(channel_stats(variance), 1e-4))), posterior.dtype());
}

Tensor unpack_latents(const Tensor& packed, const Tensor& mean, const Tensor& variance) {
  activation(packed, 4);
  check_stats(packed, mean, variance);
  const auto& s = packed.shape();
  if (s[1] != 128 || s[2] > std::numeric_limits<int>::max() / 2 || s[3] > std::numeric_limits<int>::max() / 2) {
    throw std::invalid_argument("Invalid FLUX.2 packed latent dimensions");
  }
  DeviceScope owner(packed.backend(), packed.device());
  auto x = add(mul(cast(packed, DType::Float32), sqrt(add(channel_stats(variance), 1e-4))), channel_stats(mean));
  x = permute(reshape(cast(x, packed.dtype()), {s[0], 32, 2, 2, s[2], s[3]}), {0, 1, 4, 2, 5, 3});
  return reshape(x, {s[0], 32, s[2] * 2, s[3] * 2});
}

Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale) {
  return dispatch_tensor("flux_modulate", x.shape(), {x, shift, scale}, {}, x.dtype());
}

Tensor timestep_embedding(const Tensor& times) {
  activation(times, 1);
  DeviceScope owner(times.backend(), times.device());
  auto phases = mul(unsqueeze(cast(mul(times, 1000.0), DType::Float32), -1), frequencies(128, 10000, times.backend()));
  return cast(concat({cos(phases), sin(phases)}, -1), times.dtype());
}

Tensor attention(const Tensor& query, const Tensor& key, const Tensor& value) {
  return dispatch_tensor("flux_attention", query.shape(), {query, key, value}, {}, query.dtype());
}

void detail::validate_kernel(const std::string& op, const BufferPtr& out, const std::vector<BufferPtr>& inputs,
                             const KernelParams& p) {
  auto require = [&](bool valid) {
    if (!valid) {
      throw std::invalid_argument("Invalid " + op + " operands");
    }
  };
  const bool rotary = op == "flux_rotary", mod = op == "flux_modulate";
  const int arity = rotary ? 2 : 3, rank = mod ? 3 : 4;
  require(p.output_rank() == rank && is_compute_float(out->dtype) && p.output_count() == int(out->count));
  require(inputs.size() >= size_t(arity));
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i >= size_t(arity)) {
      require(!inputs[i]);
      continue;
    }
    const auto slot = KernelInput(i);
    const int dims = rotary && i == 1 ? 3 : rank;
    require(inputs[i] && p.input_rank(slot) == dims && inputs[i]->device_ordinal() == out->device_ordinal() &&
            inputs[i]->dtype == (rotary && i == 1 ? DType::Float32 : out->dtype));
    int count = 1;
    for (int d = dims - 1; d >= 0; --d) {
      const int length = p.input_shape(slot, d);
      require(length >= 0 && (!length || count <= INT32_MAX / length) && p.input_stride(slot, d) == count);
      count *= length;
      if (i == 0) {
        require(length == p.output_shape(d));
      } else if (rotary) {
        require(length == p.output_shape(d == 0 ? 0 : d + 1) || (d == 0 && length == 1));
      } else if (mod) {
        require((length == p.output_shape(d) || (d == 1 && length == 1)) &&
                (i == 1 || length == p.input_shape(KernelInput::B, d)));
      } else if (d == 2) {
        require(length > 0 && (i == 1 || length == p.input_shape(KernelInput::B, d)));
      } else {
        require(length == p.output_shape(d));
      }
    }
    require(size_t(count) == inputs[i]->count && (i != 0 || count == p.output_count()));
    if (out->bytes && inputs[i]->bytes) {
      const auto a = reinterpret_cast<uintptr_t>(out->raw_pointer()),
                 b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
      require(a <= b ? b - a >= out->bytes : a - b >= inputs[i]->bytes);
    }
  }
  require(mod ? p.output_shape(2) > 0 : p.output_shape(3) == 128);
}
}  // namespace elman::flux2
