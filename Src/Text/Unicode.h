#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace elman {

std::u16string utf16(const std::string& text, bool replace_invalid = false);
std::string utf8(const std::u16string& text);
std::string normalize_nfc(const std::string& text);
std::string lowercase_unicode(const std::string& text);
const std::array<char16_t, 256>& bytelevel_codepoints();
std::string replace_invalid_utf8(const std::string& text);
std::string unicode_version();

class UnicodeRegexSplit {
public:
    explicit UnicodeRegexSplit(const std::string& pattern);
    ~UnicodeRegexSplit();
    UnicodeRegexSplit(UnicodeRegexSplit&&) noexcept;
    UnicodeRegexSplit& operator=(UnicodeRegexSplit&&) noexcept;
    UnicodeRegexSplit(const UnicodeRegexSplit&) = delete;
    UnicodeRegexSplit& operator=(const UnicodeRegexSplit&) = delete;
    std::vector<std::string> split(const std::string& text, bool include_unmatched = true) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace elman
