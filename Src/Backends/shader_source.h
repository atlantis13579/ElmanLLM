#pragma once
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace elman {
inline std::filesystem::path shader_executable_directory() {
#if defined(_WIN32)
    std::vector<wchar_t> path(512);
    while (path.size() <= 32768) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (length == 0) return {};
        if (length < path.size()) return std::filesystem::path(std::wstring(path.data(), length)).parent_path();
        path.resize(path.size() * 2);
    }
#elif defined(__APPLE__)
    uint32_t size = 1024;
    std::vector<char> path(size);
    if (_NSGetExecutablePath(path.data(), &size) != 0) {
        path.resize(size);
        if (_NSGetExecutablePath(path.data(), &size) != 0) return {};
    }
    std::error_code error;
    auto resolved = std::filesystem::weakly_canonical(std::filesystem::path(path.data()), error);
    if (!error) return resolved.parent_path();
    return std::filesystem::path(path.data()).parent_path();
#endif
    return {};
}

inline std::filesystem::path runtime_shader_path(const std::string& backend, const std::string& filename) {
    const auto executable = shader_executable_directory();
    if (!executable.empty()) {
        auto packaged = executable / "Shaders" / "Src" / "Operators" / backend / filename;
        if (std::filesystem::is_regular_file(packaged)) return packaged;
    }
    return std::filesystem::path(ELMAN_SOURCE_DIR) / "Src" / "Operators" / backend / filename;
}

// Resolve local includes before runtime compilation on either shader platform.
inline std::string load_shader_source(const std::filesystem::path& path, int depth = 0) {
    if (depth > 16) throw std::runtime_error("Shader include nesting is too deep");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read shader: " + path.string());
    std::ostringstream source;
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t");
        if (first != std::string::npos && line.compare(first, 8, "#include") == 0) {
            const auto opening = line.find('"', first + 8);
            if (opening != std::string::npos) {
                const auto closing = line.find('"', opening + 1);
                if (closing == std::string::npos) throw std::runtime_error("Malformed shader include");
                source << load_shader_source(path.parent_path() / line.substr(opening + 1, closing - opening - 1), depth + 1) << '\n';
                continue;
            }
        }
        source << line << '\n';
    }
    return source.str();
}
}
