#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "IO/BinaryInput.h"

namespace elman::io {

class ZipArchive {
public:
    struct Entry { uint64_t offset, compressed, size; uint32_t checksum; uint16_t method; };
    explicit ZipArchive(const std::filesystem::path& path);
    const std::map<std::string, Entry>& entries() const { return entries_; }
    std::vector<uint8_t> read(const std::string& name);

private:
    BinaryInput input_;
    uint64_t directory_ = 0;
    std::map<std::string, Entry> entries_;
};

}  // namespace elman::io
