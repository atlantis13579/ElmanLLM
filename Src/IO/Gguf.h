#pragma once

#include "IO/Safetensors.h"

namespace elman {

class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path& path);
    const std::map<std::string, SafetensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& strings() const { return strings_; }
    const std::map<std::string, double>& numbers() const { return numbers_; }
    std::vector<uint8_t> read(const std::string& name) const;
    Tensor load(const std::string& name, Backend backend) const;

private:
    std::filesystem::path path_;
    std::map<std::string, SafetensorInfo> tensors_;
    std::map<std::string, uint32_t> types_;
    std::map<std::string, std::string> strings_;
    std::map<std::string, double> numbers_;
    uint64_t data_start_ = 0;
};

}  // namespace elman
