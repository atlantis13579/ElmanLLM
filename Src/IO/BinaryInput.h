#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace elman::io {

inline void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}

inline void require(bool valid, const std::string& message) {
    if (!valid) throw std::invalid_argument(message);
}

struct Bytes {
    const uint8_t* data = nullptr;
    size_t size = 0;

    Bytes sub(size_t start, size_t count) const {
        require(start <= size && count <= size - start, "Truncated binary model data");
        return {start ? data + start : data, count};
    }
    uint64_t le(size_t start, size_t width) const {
        require(width <= sizeof(uint64_t), "Binary model integer width exceeds 64 bits");
        const auto field = sub(start, width);
        uint64_t value = 0;
        for (size_t i = 0; i < width; ++i) value |= uint64_t(field.data[i]) << (8 * i);
        return value;
    }
    std::string text(size_t start, size_t count) const {
        if (!count) return {};
        return {reinterpret_cast<const char*>(sub(start, count).data), count};
    }
};

inline Bytes view(const std::vector<uint8_t>& bytes) { return {bytes.data(), bytes.size()}; }

class BinaryInput {
public:
    explicit BinaryInput(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
        if (!stream_) throw std::runtime_error("Cannot open binary model file: " + path.u8string());
        stream_.seekg(0, std::ios::end);
        const auto end = stream_.tellg();
        if (end < 0) throw std::runtime_error("Cannot read binary file size");
        size_ = static_cast<uint64_t>(end);
        seek(0);
    }
    uint64_t size() const { return size_; }
    uint64_t position() const { return offset_; }
    uint64_t remaining() const { return size_ - offset_; }
    void seek(uint64_t offset) {
        require(offset <= size_, "Binary model offset is out of range");
        stream_.seekg(static_cast<std::streamoff>(offset));
        if (!stream_) throw std::runtime_error("Cannot seek binary model file");
        offset_ = offset;
    }
    void skip(uint64_t distance) {
        require(distance <= remaining(), "Truncated binary model field");
        seek(offset_ + distance);
    }
    void read(void* destination, size_t count) {
        require(count <= remaining() && count <= uint64_t(std::numeric_limits<std::streamsize>::max()), "Truncated binary model field");
        if (!count) return;
        stream_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (!stream_) throw std::runtime_error("Cannot read binary model file");
        offset_ += count;
    }
    std::vector<uint8_t> bytes(uint64_t count) {
        require(count <= remaining() && count <= std::numeric_limits<size_t>::max(), "Truncated binary model block");
        std::vector<uint8_t> data(static_cast<size_t>(count));
        read(data.data(), data.size());
        return data;
    }
    std::string text(uint64_t count) {
        require(count <= remaining() && count <= std::numeric_limits<size_t>::max(), "Truncated binary model string");
        std::string data(static_cast<size_t>(count), '\0');
        read(data.data(), data.size());
        return data;
    }
    uint64_t le(size_t width) {
        uint8_t field[8];
        require(width <= sizeof(field), "Binary model integer width exceeds 64 bits");
        read(field, width);
        return Bytes{field, width}.le(0, width);
    }

private:
    std::ifstream stream_;
    uint64_t size_ = 0, offset_ = 0;
};

}  // namespace elman::io
