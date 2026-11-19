#include "ToolIO.h"

#include <array>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "IO/AtomicFile.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#elif defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#else
#include <openssl/evp.h>
#endif

namespace elman::tools {
namespace {

#ifdef _WIN32
struct Sha256 {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::vector<unsigned char> storage;
    ~Sha256() {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    }
};
#endif

void check_hash(bool ok) {
    if (!ok) throw std::runtime_error("Cannot compute SHA-256");
}

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for hashing: " + path.u8string());
#ifdef _WIN32
    Sha256 state;
    DWORD size = 0, written = 0;
    check_hash(BCryptOpenAlgorithmProvider(&state.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0);
    check_hash(BCryptGetProperty(state.algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&size),
                                 sizeof(size), &written, 0) >= 0);
    state.storage.resize(size);
    check_hash(BCryptCreateHash(state.algorithm, &state.hash, state.storage.data(), size, nullptr, 0, 0) >= 0);
#elif defined(__APPLE__)
    CC_SHA256_CTX hash{};
    check_hash(CC_SHA256_Init(&hash) == 1);
#else
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    check_hash(hash && EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) == 1);
#endif
    std::vector<unsigned char> buffer(1024 * 1024);
    std::array<unsigned char, 32> digest{};
    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        if (const auto count = file.gcount()) {
#ifdef _WIN32
            check_hash(BCryptHashData(state.hash, buffer.data(), static_cast<ULONG>(count), 0) >= 0);
#elif defined(__APPLE__)
            check_hash(CC_SHA256_Update(&hash, buffer.data(), static_cast<CC_LONG>(count)) == 1);
#else
            check_hash(EVP_DigestUpdate(hash.get(), buffer.data(), static_cast<size_t>(count)) == 1);
#endif
        }
    }
    if (!file.eof() || file.bad()) throw std::runtime_error("Cannot read file for hashing: " + path.u8string());
#ifdef _WIN32
    check_hash(BCryptFinishHash(state.hash, digest.data(), static_cast<ULONG>(digest.size()), 0) >= 0);
#elif defined(__APPLE__)
    check_hash(CC_SHA256_Final(digest.data(), &hash) == 1);
#else
    unsigned int size = 0;
    check_hash(EVP_DigestFinal_ex(hash.get(), digest.data(), &size) == 1 && size == digest.size());
#endif
    std::string hex;
    hex.reserve(64);
    for (auto byte : digest) {
        constexpr char digits[] = "0123456789abcdef";
        hex += digits[byte >> 4];
        hex += digits[byte & 15];
    }
    return hex;
}

std::filesystem::path executable_path() {
#ifdef _WIN32
    std::wstring path(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (!length || length >= path.size()) throw std::runtime_error("Cannot determine executable path");
    path.resize(length);
    return path;
#elif defined(__APPLE__)
    uint32_t length = 0;
    _NSGetExecutablePath(nullptr, &length);
    std::vector<char> path(length);
    if (_NSGetExecutablePath(path.data(), &length)) throw std::runtime_error("Cannot determine executable path");
    return std::filesystem::canonical(path.data());
#else
    return std::filesystem::canonical("/proc/self/exe");
#endif
}

void write_json(std::ostream& stream, const Json& value) {
    stream << value.dump(2) << '\n';
    stream.flush();
    if (!stream) throw std::runtime_error("Cannot write JSON");
}

void write_json_atomically(const std::filesystem::path& path, const Json& value) {
    write_file_atomically(path, [&](const auto& temporary) {
        std::ofstream stream(temporary, std::ios::binary);
        write_json(stream, value);
        stream.close();
        if (!stream) throw std::runtime_error("Cannot close JSON file: " + path.u8string());
    });
}

}  // namespace elman::tools
