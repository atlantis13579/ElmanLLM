#pragma once

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace elman::tools {

inline int integer_argument(std::string_view text, std::string_view option) {
    int value = 0;
    const char* const end = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end)
        throw std::invalid_argument("Invalid integer for " + std::string(option));
    return value;
}

#ifdef _WIN32
inline std::string utf8_argument(const wchar_t* text) {
    const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, -1, nullptr, 0, nullptr, nullptr);
    if (!length) throw std::invalid_argument("Invalid Windows argument");
    std::string value(static_cast<size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, -1, value.data(), length, nullptr, nullptr);
    value.pop_back();
    return value;
}
#else
inline std::string utf8_argument(const char* text) { return text; }
#endif

}  // namespace elman::tools
