#pragma once

#include "Core/backend.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace elman {
inline std::vector<std::uint32_t> encode_words(const std::vector<float>& values, DType dtype) {
    std::vector<std::uint32_t> words(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (dtype == DType::Float32) {
            std::memcpy(&words[i], &values[i], sizeof(float));
        } else if (dtype == DType::Bool) {
            words[i] = values[i] != 0.0f ? 1u : 0u;
        } else {
            const double value = values[i];
            if (std::isnan(value)) words[i] = 0u;
            else if (value >= std::numeric_limits<std::int32_t>::max()) words[i] = 2147483647u;
            else if (value <= std::numeric_limits<std::int32_t>::min()) words[i] = 2147483648u;
            else words[i] = static_cast<std::uint32_t>(static_cast<std::int32_t>(value));
        }
    }
    return words;
}

inline std::vector<float> decode_words(const std::vector<std::uint32_t>& words, DType dtype) {
    std::vector<float> values(words.size());
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (dtype == DType::Float32) std::memcpy(&values[i], &words[i], sizeof(float));
        else if (dtype == DType::Int32) values[i] = static_cast<float>(static_cast<std::int32_t>(words[i]));
        else values[i] = words[i] != 0 ? 1.0f : 0.0f;
    }
    return values;
}
}
