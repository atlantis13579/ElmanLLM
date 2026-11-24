#pragma once

#include "IO/Safetensors.h"
#include "Core/Quantization.h"

namespace elman {

// Returns an explicit GGUF file or the sole GGUF in a model directory.
std::filesystem::path find_gguf_model(const std::filesystem::path& location);

class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path& path);
    const std::map<std::string, SafetensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& strings() const { return strings_; }
    const std::map<std::string, double>& numbers() const { return numbers_; }
    const std::map<std::string, std::vector<std::string>>& string_arrays() const { return string_arrays_; }
    const std::map<std::string, std::vector<double>>& number_arrays() const { return number_arrays_; }
    uint32_t encoding_type(const std::string& name) const { return types_.at(name); }
    std::vector<uint8_t> read(const std::string& name) const;
    Tensor load(const std::string& name, Backend backend) const;
    QuantizedTensor load_quantized(const std::string& name, Backend backend) const;

private:
    std::filesystem::path path_;
    std::map<std::string, SafetensorInfo> tensors_;
    std::map<std::string, uint32_t> types_;
    std::map<std::string, std::string> strings_;
    std::map<std::string, double> numbers_;
    std::map<std::string, std::vector<std::string>> string_arrays_;
    std::map<std::string, std::vector<double>> number_arrays_;
    uint64_t data_start_ = 0;
};

}  // namespace elman
