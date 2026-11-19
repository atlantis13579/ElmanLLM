#pragma once

#include "Core/Tensor.h"
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <map>
#include <string>

namespace elman {

struct SafetensorInfo {
    DType dtype;
    Shape shape;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct SafetensorSpec {
    std::string name;
    DType dtype;
    Shape shape;
};

// Payloads follow the header in specification order.
void write_safetensors_header(std::ostream& stream, const std::vector<SafetensorSpec>& entries,
                              const std::map<std::string, std::string>& metadata = {});
void save_safetensors(const std::filesystem::path& path, const std::map<std::string, Tensor>& tensors);

// Validates the complete catalog without allocating tensor storage. Loading a
// tensor reads only its payload, preserving integer and floating-point bits.
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::filesystem::path& path);
    const std::map<std::string, SafetensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    const SafetensorInfo& info(const std::string& name) const;
    void read_bytes(const std::string& name, uint64_t offset, void* output, size_t count) const;
    Tensor load(const std::string& name, Backend backend = Backend::CPU) const;
    Tensor load_slice(const std::string& name, int axis, int offset, int count, Backend backend) const;
    // Explicit byte-preserving load for a separately validated logical catalog.
    Tensor load_storage(const std::string& name, const Shape& shape, DType dtype, Backend backend) const;

private:
    std::filesystem::path path_;
    uint64_t data_start_ = 0;
    uint64_t file_size_ = 0;
    std::map<std::string, SafetensorInfo> tensors_;
    std::map<std::string, std::string> metadata_;
};

// Hugging Face single-file and sharded safetensors checkpoints. Shard names
// must be relative files within the supplied model directory.
class SafetensorsWeights {
public:
    explicit SafetensorsWeights(const std::filesystem::path& directory);
    const std::map<std::string, SafetensorInfo>& tensors() const { return tensors_; }
    const SafetensorInfo& info(const std::string& name) const;
    void read_bytes(const std::string& name, uint64_t offset, void* output, size_t count) const;
    Tensor load(const std::string& name, Backend backend = Backend::CPU) const;
    Tensor load_slice(const std::string& name, int axis, int offset, int count, Backend backend) const;
    // Explicit byte-preserving load for a separately validated logical catalog.
    Tensor load_storage(const std::string& name, const Shape& shape, DType dtype, Backend backend) const;

private:
    std::vector<SafetensorsFile> files_;
    std::map<std::string, size_t> locations_;
    std::map<std::string, SafetensorInfo> tensors_;
};

} // namespace elman
