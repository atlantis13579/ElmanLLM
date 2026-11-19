#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Core/Backend.h"
#include "Core/Float16.h"
#include "Core/LowPrecisionFloat.h"
#include "Core/PackedInteger.h"

namespace elman {
inline std::vector<std::uint32_t> encode_words(const std::vector<float>& values,
                                               DType dtype) {
  if (dtype != DType::Float32 && dtype != DType::Int32 && dtype != DType::Bool)
    throw std::invalid_argument("Float upload conversion is not implemented for " + std::string(dtype_name(dtype)));
  std::vector<std::uint32_t> words(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (dtype == DType::Float32) {
      std::memcpy(&words[i], &values[i], sizeof(float));
    } else if (dtype == DType::Bool) {
      words[i] = values[i] != 0.0f ? 1u : 0u;
    } else {
      const double value = values[i];
      if (std::isnan(value)) {
        words[i] = 0u;
      } else if (value >= std::numeric_limits<std::int32_t>::max()) {
        words[i] = 2147483647u;
      } else if (value <= std::numeric_limits<std::int32_t>::min()) {
        words[i] = 2147483648u;
      } else {
        words[i] = static_cast<std::uint32_t>(static_cast<std::int32_t>(value));
      }
    }
  }
  return words;
}

inline std::vector<float> decode_words(const std::vector<std::uint32_t>& words,
                                       DType dtype) {
  if (dtype != DType::Float32 && dtype != DType::Int32 && dtype != DType::Bool)
    throw std::invalid_argument("Float download conversion is not implemented for " + std::string(dtype_name(dtype)));
  std::vector<float> values(words.size());
  for (std::size_t i = 0; i < words.size(); ++i) {
    if (dtype == DType::Float32) {
      std::memcpy(&values[i], &words[i], sizeof(float));
    } else if (dtype == DType::Int32) {
      values[i] = static_cast<float>(static_cast<std::int32_t>(words[i]));
    } else {
      values[i] = words[i] != 0 ? 1.0f : 0.0f;
    }
  }
  return values;
}
inline std::vector<uint8_t> encode_host_values(const std::vector<float>& values, DType dtype) {
  if (is_packed_integer(dtype)) {
    std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, values.size()));
    const auto read = [&](uint64_t i) -> uint64_t {
      const double value = values[size_t(i)];
      const double bound = packed_integer_signed(dtype) ? 2147483648.0 : 9223372036854775808.0;
      return std::isfinite(value) && value >= -bound && value < bound ? uint64_t(int64_t(value)) : 0;
    };
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = packed_integer_byte(i, values.size(), packed_integer_bits(dtype), read);
    return bytes;
  }
  if (is_low_precision_float(dtype)) {
    std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, values.size()), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      if (is_float8(dtype)) bytes[i] = numeric::float_to_float8(values[i], dtype);
      else bytes[i / 2] |= uint8_t(numeric::double_to_float4(double(values[i])) << ((i & 1) * 4));
    }
    return bytes;
  }
  if (dtype == DType::Float64) {
    std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, values.size()));
    for (size_t i = 0; i < values.size(); ++i) {
      const double value = values[i];
      std::memcpy(bytes.data() + sizeof(double) * i, &value, sizeof(value));
    }
    return bytes;
  }
  if (is_narrow_float(dtype)) {
    std::vector<uint8_t> bytes(values.size() * 2);
    for (size_t i = 0; i < values.size(); ++i) {
      uint16_t bits = dtype == DType::Float16 ? numeric::float_to_half(values[i]) : numeric::float_to_bfloat(values[i]);
      std::memcpy(bytes.data() + 2 * i, &bits, 2);
    }
    return bytes;
  }
  auto words = encode_words(values, dtype);
  std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, values.size()));
  if (dtype == DType::Bool) {
    for (size_t i = 0; i < words.size(); ++i) bytes[i] = uint8_t(words[i]);
  } else if (!bytes.empty()) std::memcpy(bytes.data(), words.data(), bytes.size());
  return bytes;
}
inline std::vector<float> decode_host_values(const std::vector<uint8_t>& bytes, DType dtype, size_t count) {
  if (is_packed_integer(dtype)) {
    if (bytes.size() != dtype_storage_bytes(dtype, count)) throw std::invalid_argument("Host decode size mismatch");
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = float(packed_integer_load(bytes.data(), i, dtype));
    return values;
  }
  if (is_low_precision_float(dtype)) {
    if (bytes.size() != dtype_storage_bytes(dtype, count)) throw std::invalid_argument("Host decode size mismatch");
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
      values[i] = is_float8(dtype) ? numeric::float8_to_float(bytes[i], dtype) :
          float(numeric::float4_to_double(uint8_t(bytes[i / 2] >> ((i & 1) * 4))));
    return values;
  }
  if (dtype == DType::Float64) {
    if (bytes.size() != dtype_storage_bytes(dtype, count)) throw std::invalid_argument("Host decode size mismatch");
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
      double value;
      std::memcpy(&value, bytes.data() + sizeof(double) * i, sizeof(value));
      values[i] = float(value);
    }
    return values;
  }
  if (is_narrow_float(dtype)) {
    if (bytes.size() != dtype_storage_bytes(dtype, count)) throw std::invalid_argument("Host decode size mismatch");
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + 2 * i, 2);
      values[i] = dtype == DType::Float16 ? numeric::half_to_float(bits) : numeric::bfloat_to_float(bits);
    }
    return values;
  }
  if (dtype != DType::Float32 && dtype != DType::Int32 && dtype != DType::Bool)
    throw std::invalid_argument("Float download conversion is not implemented for " + std::string(dtype_name(dtype)));
  if (bytes.size() != dtype_storage_bytes(dtype, count)) throw std::invalid_argument("Host decode size mismatch");
  std::vector<uint32_t> words(count);
  if (dtype == DType::Bool) {
    for (size_t i = 0; i < count; ++i) words[i] = bytes[i] != 0;
  } else if (!bytes.empty()) std::memcpy(words.data(), bytes.data(), bytes.size());
  return decode_words(words, dtype);
}
inline std::vector<uint8_t> encode_host_integers(const std::vector<int32_t>& values, DType dtype) {
  if (is_packed_integer(dtype)) {
    std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, values.size()));
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = packed_integer_byte(i, values.size(), packed_integer_bits(dtype),
                                     [&](uint64_t index) { return values[size_t(index)]; });
    return bytes;
  }
  if (dtype != DType::Int32 && dtype != DType::Bool) throw std::invalid_argument("Int32 upload requires int32/bool storage");
  std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, values.size()));
  if (dtype == DType::Bool) {
    for (size_t i = 0; i < values.size(); ++i) bytes[i] = values[i] != 0;
  } else if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
inline std::vector<int32_t> decode_host_integers(const std::vector<uint8_t>& bytes, DType dtype, size_t count) {
  if (is_packed_integer(dtype)) {
    if (bytes.size() != dtype_storage_bytes(dtype, count)) throw std::invalid_argument("Host integer decode size mismatch");
    std::vector<int32_t> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = packed_integer_load(bytes.data(), i, dtype);
    return values;
  }
  if (dtype != DType::Int32 && dtype != DType::Bool) throw std::invalid_argument("Int32 download requires int32/bool storage");
  if (bytes.size() != dtype_storage_bytes(dtype, count)) throw std::invalid_argument("Host integer decode size mismatch");
  std::vector<int32_t> values(count);
  if (dtype == DType::Bool) {
    for (size_t i = 0; i < count; ++i) values[i] = bytes[i] != 0;
  } else if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}
}  // namespace elman
