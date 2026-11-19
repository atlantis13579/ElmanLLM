#include "Text/Unicode.h"

#ifdef _WIN32
#include <icu.h>
#else
#include <unicode/uchar.h>
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#else
#include <unicode/unorm2.h>
#endif
#include <unicode/uregex.h>
#include <unicode/ustring.h>
#endif

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace elman {
namespace {
void check(UErrorCode code, const char* operation) {
    if (U_FAILURE(code)) throw std::runtime_error(std::string(operation) + ": " + u_errorName(code));
}
int32_t length(size_t size) {
    if (size > size_t(std::numeric_limits<int32_t>::max())) throw std::invalid_argument("Unicode text is too long");
    return static_cast<int32_t>(size);
}
struct RegexCloser { void operator()(URegularExpression* regex) const { if (regex) uregex_close(regex); } };
using Regex = std::unique_ptr<URegularExpression, RegexCloser>;
}

std::u16string utf16(const std::string& text, bool replace_invalid) {
    static_assert(sizeof(UChar) == sizeof(char16_t), "ICU requires 16-bit UTF-16 code units");
    int32_t required = 0;
    UErrorCode status = U_ZERO_ERROR;
    auto convert = [&](UChar* output, int32_t capacity) {
        if (replace_invalid)
            u_strFromUTF8WithSub(output, capacity, &required, text.data(), length(text.size()), 0xfffd, nullptr, &status);
        else u_strFromUTF8(output, capacity, &required, text.data(), length(text.size()), &status);
    };
    convert(nullptr, 0);
    if (status != U_BUFFER_OVERFLOW_ERROR) check(status, "Decode UTF-8");
    status = U_ZERO_ERROR;
    std::u16string result(size_t(required), u'\0');
    convert(reinterpret_cast<UChar*>(result.data()), required);
    check(status, "Decode UTF-8");
    return result;
}

std::string utf8(const std::u16string& text) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t required = 0;
    u_strToUTF8(nullptr, 0, &required, reinterpret_cast<const UChar*>(text.data()), length(text.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) check(status, "Encode UTF-8");
    status = U_ZERO_ERROR;
    std::string result(size_t(required), '\0');
    u_strToUTF8(result.data(), required, &required, reinterpret_cast<const UChar*>(text.data()), length(text.size()), &status);
    check(status, "Encode UTF-8");
    return result;
}

std::string normalize_nfc(const std::string& text) {
    auto source = utf16(text);
#ifdef __APPLE__
    using String = std::unique_ptr<std::remove_pointer_t<CFMutableStringRef>, decltype(&CFRelease)>;
    String normalizer(CFStringCreateMutable(kCFAllocatorDefault, 0), CFRelease);
    if (!normalizer) throw std::runtime_error("Create NFC normalizer failed");
    CFStringAppendCharacters(normalizer.get(), reinterpret_cast<const UniChar*>(source.data()), source.size());
    CFStringNormalize(normalizer.get(), kCFStringNormalizationFormC);
    const auto count = CFStringGetLength(normalizer.get());
    std::u16string normalized(size_t(count), u'\0');
    CFStringGetCharacters(normalizer.get(), CFRangeMake(0, count), reinterpret_cast<UniChar*>(normalized.data()));
    return utf8(normalized);
#else
    UErrorCode status = U_ZERO_ERROR;
    auto normalizer = unorm2_getNFCInstance(&status);
    check(status, "Create NFC normalizer");
    int32_t required = unorm2_normalize(normalizer, reinterpret_cast<const UChar*>(source.data()), length(source.size()), nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) check(status, "Normalize NFC");
    status = U_ZERO_ERROR;
    std::u16string result(size_t(required), u'\0');
    unorm2_normalize(normalizer, reinterpret_cast<const UChar*>(source.data()), length(source.size()),
                     reinterpret_cast<UChar*>(result.data()), required, &status);
    check(status, "Normalize NFC");
    return utf8(result);
#endif
}

std::string replace_invalid_utf8(const std::string& text) { return utf8(utf16(text, true)); }

std::string lowercase_unicode(const std::string& text) {
    const auto source = utf16(text);
    UErrorCode status = U_ZERO_ERROR;
    auto lower = [&](UChar* output, int32_t capacity) {
        return u_strToLower(output, capacity, reinterpret_cast<const UChar*>(source.data()), length(source.size()), "", &status);
    };
    const auto required = lower(nullptr, 0);
    if (status != U_BUFFER_OVERFLOW_ERROR) check(status, "Measure Unicode lowercase");
    status = U_ZERO_ERROR;
    std::u16string result(size_t(required), u'\0');
    lower(reinterpret_cast<UChar*>(result.data()), required);
    check(status, "Convert Unicode lowercase");
    return utf8(result);
}

const std::array<char16_t, 256>& bytelevel_codepoints() {
    static const auto codes = [] {
        std::array<char16_t, 256> result{};
        int next = 256;
        for (int byte = 0; byte < 256; ++byte) {
            const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            result[byte] = static_cast<char16_t>(visible ? byte : next++);
        }
        return result;
    }();
    return codes;
}

std::string unicode_version() {
    UVersionInfo version;
    u_getUnicodeVersion(version);
    char result[U_MAX_VERSION_STRING_LENGTH];
    u_versionToString(version, result);
    return result;
}

struct UnicodeRegexSplit::Impl { Regex regex; };
UnicodeRegexSplit::UnicodeRegexSplit(const std::string& pattern) : impl_(std::make_unique<Impl>()) {
    auto source = utf16(pattern);
    UErrorCode status = U_ZERO_ERROR;
    UParseError error{};
    impl_->regex.reset(uregex_open(reinterpret_cast<const UChar*>(source.data()), length(source.size()), 0, &error, &status));
    check(status, "Compile tokenizer regular expression");
}
UnicodeRegexSplit::~UnicodeRegexSplit() = default;
UnicodeRegexSplit::UnicodeRegexSplit(UnicodeRegexSplit&&) noexcept = default;
UnicodeRegexSplit& UnicodeRegexSplit::operator=(UnicodeRegexSplit&&) noexcept = default;

std::vector<std::string> UnicodeRegexSplit::split(const std::string& text, bool include_unmatched) const {
    auto source = utf16(text);
    UErrorCode status = U_ZERO_ERROR;
    Regex regex(uregex_clone(impl_->regex.get(), &status));
    check(status, "Clone tokenizer regular expression");
    uregex_setText(regex.get(), reinterpret_cast<const UChar*>(source.data()), length(source.size()), &status);
    check(status, "Set tokenizer regular expression text");
    std::vector<std::string> result;
    int32_t previous = 0;
    while (uregex_findNext(regex.get(), &status)) {
        int32_t begin = uregex_start(regex.get(), 0, &status), end = uregex_end(regex.get(), 0, &status);
        check(status, "Find tokenizer split");
        if (begin == end) throw std::invalid_argument("Tokenizer regular expression matched an empty string");
        if (include_unmatched && begin > previous) result.push_back(utf8(source.substr(previous, begin - previous)));
        result.push_back(utf8(source.substr(begin, end - begin)));
        previous = end;
    }
    check(status, "Find tokenizer split");
    if (include_unmatched && size_t(previous) < source.size()) result.push_back(utf8(source.substr(previous)));
    return result;
}

} // namespace elman
