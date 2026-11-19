#pragma once

#include "Core/Tensor.h"
#include "IO/Json.h"

#include <stdexcept>

namespace elman::fixture {
inline Shape shape(const Json& value) {
    Shape result;
    for (const auto& dimension : value.array()) result.push_back(dimension.integer());
    return result;
}

inline Tensor input(const Json& spec, Backend backend) {
    if (spec.is_null() || spec.bool_or("undefined", false)) return {};
    const bool compact = spec.contains("bytes");
    auto nibble = [](char digit) -> uint8_t {
        if (digit >= '0' && digit <= '9') return uint8_t(digit - '0');
        if (digit >= 'a' && digit <= 'f') return uint8_t(digit - 'a' + 10);
        throw std::runtime_error("Invalid hexadecimal tensor fixture");
    };
    const auto hex = spec.at(compact ? "bytes" : "raw_hex").string();
    if (hex.size() % 2) throw std::runtime_error("Incomplete hexadecimal tensor fixture");
    std::vector<uint8_t> value(hex.size()/2);
    for (size_t i = 0; i < value.size(); ++i) value[i] = uint8_t((nibble(hex[2*i]) << 4) | nibble(hex[2*i+1]));
    return Tensor::from_raw(shape(spec.at("shape")), DType(spec.at(compact ? "type" : "dtype").integer()), value, backend);
}
} // namespace elman::fixture
