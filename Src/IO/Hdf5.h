#pragma once

#include <map>

#include "Core/Tensor.h"
#include "IO/BinaryInput.h"

namespace elman::io {

// HDF5 object metadata and dense dataset storage used by Keras exporters.
class Hdf5File {
public:
    explicit Hdf5File(const std::filesystem::path& path);
    explicit Hdf5File(std::vector<uint8_t> bytes);
    Hdf5File(const Hdf5File&) = delete;
    Hdf5File& operator=(const Hdf5File&) = delete;
    bool contains(const std::string& path);
    std::vector<std::string> children(const std::string& path);
    std::string attribute_text(const std::string& path, const std::string& name, bool required = true);
    Tensor tensor(const std::string& path, Backend backend);

private:
    struct Object {
        std::multimap<uint16_t, Bytes> messages;
        std::map<std::string, uint64_t> children;
    };
    uint64_t lookup(const std::string& path);
    const Object& object(uint64_t offset);
    Bytes absolute(uint64_t offset, uint64_t size) const;
    Bytes remainder(uint64_t offset) const;
    uint64_t address(Bytes bytes, size_t start) const;
    bool defined(uint64_t address) const;
    void header_chunk(Object& object, Bytes chunk, bool legacy, bool order, unsigned depth) const;
    void symbol_tree(Object& object, uint64_t node, Bytes heap, unsigned depth) const;
    void inline_link(Object& object, Bytes data) const;
    std::string heap_string(uint64_t address, uint32_t index, uint64_t length) const;
    std::vector<uint8_t> chunked(const Object& object, Bytes layout, const Shape& shape, size_t width) const;

    std::vector<uint8_t> data_;
    std::map<uint64_t, Object> objects_;
    size_t address_width_ = 0, size_width_ = 0;
    uint64_t base_ = 0, root_ = 0;
};

}  // namespace elman::io
