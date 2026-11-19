#include "IO/Hdf5.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <string_view>

#include "Core/TensorDispatch.h"
#include "IO/Deflate.h"

namespace elman::io {
namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    BinaryInput file(path);
    return file.bytes(file.size());
}

size_t align8(size_t count) {
    require(count <= std::numeric_limits<size_t>::max() - 7, "HDF5 message size overflow");
    return (count + 7) & ~size_t(7);
}

std::string cstring(Bytes bytes) {
    if (!bytes.size) return {};
    const auto* zero = static_cast<const uint8_t*>(std::memchr(bytes.data, 0, bytes.size));
    return bytes.text(0, zero ? size_t(zero - bytes.data) : bytes.size);
}

Shape shape(Bytes dataspace, size_t width) {
    const auto version = dataspace.le(0, 1), rank = dataspace.le(1, 1);
    require(rank <= 8 && (version == 1 || version == 2), "Unsupported HDF5 dataspace");
    if (version == 2) require(dataspace.le(3, 1) <= 1, "HDF5 null dataspace is unsupported");
    const int offset = version == 1 ? 8 : 4;
    Shape result;
    for (size_t i = 0; i < rank; ++i) {
        const auto dim = dataspace.le(offset + width * i, width);
        require(dim <= uint64_t(INT32_MAX), "HDF5 dimension exceeds the tensor limit");
        result.push_back(int(dim));
    }
    return result;
}

DType data_type(Bytes type) {
    const auto family = type.le(0, 1) & 15, flags = type.le(1, 3), width = type.le(4, 4);
    if (family == 1) {
        require(!(flags & 0x40) && type.le(8, 2) == 0 && type.le(10, 2) == width * 8,
                "Unsupported HDF5 floating-point representation");
        if (width == 2 && type.le(13, 1) == 5) return DType::Float16;
        if (width == 2 && type.le(13, 1) == 8) return DType::BFloat16;
        if (width == 4 && type.le(13, 1) == 8) return DType::Float32;
        if (width == 8 && type.le(13, 1) == 11) return DType::Float64;
    }
    if (family == 0 && type.le(8, 2) == 0 && type.le(10, 2) == width * 8) {
        const bool sign = (flags & 8) != 0;
        if (width == 1) return sign ? DType::Int8 : DType::UInt8;
        if (width == 2) return sign ? DType::Int16 : DType::UInt16;
        if (width == 4) return sign ? DType::Int32 : DType::UInt32;
        if (width == 8) return sign ? DType::Int64 : DType::UInt64;
    }
    throw std::invalid_argument("Unsupported HDF5 tensor datatype");
}

struct Filter { uint16_t id; std::vector<uint32_t> parameters; };

std::vector<Filter> filters(Bytes pipeline) {
    const auto version = pipeline.le(0, 1), count = pipeline.le(1, 1);
    require((version == 1 || version == 2) && count <= 32, "Unsupported HDF5 filter pipeline");
    size_t offset = version == 1 ? 8 : 2;
    std::vector<Filter> filters;
    for (size_t i = 0; i < count; ++i) {
        const auto id = uint16_t(pipeline.le(offset, 2));
        offset += 2;
        size_t name = 0;
        if (version == 1 || id >= 256) {
            name = size_t(pipeline.le(offset, 2));
            offset += 2;
        }
        const auto parameters = size_t(pipeline.le(offset + 2, 2));
        offset += 4 + name;
        Filter filter{id, {}};
        for (size_t j = 0; j < parameters; ++j)
            filter.parameters.push_back(uint32_t(pipeline.le(offset + j * 4, 4)));
        offset += (version == 1 ? align8(parameters * 4) : parameters * 4);
        filters.push_back(std::move(filter));
    }
    return filters;
}

}  // namespace

Hdf5File::Hdf5File(const std::filesystem::path& path) : Hdf5File(read_file(path)) {}

Hdf5File::Hdf5File(std::vector<uint8_t> bytes) : data_(std::move(bytes)) {
    const auto all = view(data_);
    size_t signature = 0;
    constexpr uint64_t magic = 0x0a1a0a0d46444889ULL;
    while (signature <= all.size && all.size - signature >= 12 && all.le(signature, 8) != magic) {
        if (signature > all.size / 2) break;
        signature = signature ? signature * 2 : 512;
    }
    require(signature <= all.size && all.size - signature >= 12 && all.le(signature, 8) == magic, "Invalid HDF5 signature");
    const auto super = all.sub(signature, all.size - signature);
    const auto version = super.le(8, 1);
    require(version <= 3, "Unsupported HDF5 superblock version");
    const bool legacy = version <= 1;
    address_width_ = size_t(super.le(legacy ? 13 : 9, 1));
    size_width_ = size_t(super.le(legacy ? 14 : 10, 1));
    require((address_width_ == 2 || address_width_ == 4 || address_width_ == 8) &&
            (size_width_ == 2 || size_width_ == 4 || size_width_ == 8), "Unsupported HDF5 address width");
    const size_t start = legacy ? (version == 0 ? 24 : 28) : 12;
    base_ = address(super, start);
    root_ = address(super, start + address_width_ * (legacy ? 4 : 3) + (legacy ? size_width_ : 0));
    require(base_ <= data_.size() && defined(root_), "Invalid HDF5 file base or root address");
}

bool Hdf5File::defined(uint64_t value) const {
    const auto undef = address_width_ == 8 ? UINT64_MAX : (uint64_t(1) << (address_width_ * 8)) - 1;
    return value != undef;
}

uint64_t Hdf5File::address(Bytes bytes, size_t start) const { return bytes.le(start, address_width_); }

Bytes Hdf5File::absolute(uint64_t offset, uint64_t size) const {
    require(defined(offset) && offset <= data_.size() - base_ && size <= data_.size() - base_ - offset,
            "HDF5 address is outside the file");
    return view(data_).sub(size_t(base_ + offset), size_t(size));
}

Bytes Hdf5File::remainder(uint64_t offset) const {
    require(offset <= data_.size() - base_, "HDF5 address is outside the file");
    return absolute(offset, data_.size() - base_ - offset);
}

void Hdf5File::header_chunk(Object& object, Bytes chunk, bool legacy, bool order, unsigned depth) const {
    require(depth < 24, "Too many HDF5 object continuations");
    const size_t width = (legacy ? 8 : 4) + (order ? 2 : 0);
    size_t pos = 0;
    while (chunk.size - pos >= width) {
        const auto id = uint16_t(chunk.le(pos, legacy ? 2 : 1));
        const auto length = size_t(chunk.le(pos + (legacy ? 2 : 1), 2));
        pos += width;
        const auto message = chunk.sub(pos, length);
        pos += length;
        if (id == 0) continue;
        if (id == 16) {
            auto continuation = absolute(address(message, 0), message.le(address_width_, size_width_));
            if (!legacy) {
                require(continuation.size >= 8 && continuation.text(0, 4) == "OCHK", "Invalid HDF5 continuation");
                continuation = continuation.sub(4, continuation.size - 8);
            }
            header_chunk(object, continuation, legacy, order, depth + 1);
        } else object.messages.emplace(id, message);
    }
}

void Hdf5File::symbol_tree(Object& object, uint64_t addr, Bytes heap, unsigned depth) const {
    require(depth < 64, "HDF5 group tree exceeds the importer limit");
    const auto index = remainder(addr);
    require(index.text(0, 4) == "TREE" && index.le(4, 1) == 0, "Invalid HDF5 group tree");
    const auto level = index.le(5, 1), count = index.le(6, 2);
    const size_t start = 8 + 2 * address_width_;
    for (size_t i = 0; i < count; ++i) {
        const auto child = address(index, start + i * (size_width_ + address_width_) + size_width_);
        if (level) {
            const auto branch = absolute(child, 8);
            require(branch.le(5, 1) + 1 == level, "Invalid HDF5 group tree depth");
            symbol_tree(object, child, heap, depth + 1);
            continue;
        }
        const auto leaf = remainder(child);
        require(leaf.text(0, 4) == "SNOD" && leaf.le(4, 1) == 1, "Invalid HDF5 group symbol node");
        const auto entries = leaf.le(6, 2);
        for (size_t j = 0; j < entries; ++j) {
            const size_t pos = 8 + j * (size_width_ + address_width_ + 24);
            const auto offset = size_t(leaf.le(pos, size_width_));
            const auto object_id = address(leaf, pos + size_width_);
            if (leaf.le(pos + size_width_ + address_width_, 4) == 2) continue;
            require(offset < heap.size && defined(object_id), "Invalid HDF5 group symbol entry");
            const auto name = cstring(heap.sub(offset, heap.size - offset));
            require(!name.empty() && object.children.emplace(name, object_id).second, "Duplicate HDF5 group child " + name);
        }
    }
}

void Hdf5File::inline_link(Object& object, Bytes data) const {
    require(data.le(0, 1) == 1, "Unsupported HDF5 link version");
    const auto flags = data.le(1, 1);
    size_t offset = 2;
    const auto type = flags & 8 ? data.le(offset++, 1) : 0;
    if (flags & 4) offset += 8;
    if (flags & 16) ++offset;
    const size_t name_width = 1u << (flags & 3);
    const auto length = data.le(offset, name_width);
    offset += name_width;
    require(length <= data.size - offset, "Truncated HDF5 link name");
    const auto name = data.text(offset, size_t(length));
    if (type == 0) {
        const auto target = address(data, offset + size_t(length));
        require(!name.empty() && defined(target) && object.children.emplace(name, target).second, "Invalid HDF5 hard link");
    }
}

const Hdf5File::Object& Hdf5File::object(uint64_t offset) {
    if (const auto found = objects_.find(offset); found != objects_.end()) return found->second;
    const auto header = remainder(offset);
    Object result;
    if (header.le(0, 1) == 1) {
        header_chunk(result, header.sub(16, size_t(header.le(8, 4))), true, false, 0);
    } else {
        require(header.text(0, 4) == "OHDR" && header.le(4, 1) == 2, "Unsupported HDF5 object header");
        const auto flags = header.le(5, 1);
        size_t start = 6 + ((flags & 32) ? 16 : 0) + ((flags & 16) ? 4 : 0);
        const size_t width = 1u << (flags & 3);
        const auto length = header.le(start, width);
        require(length <= header.size - start - width, "Truncated HDF5 object header");
        header_chunk(result, header.sub(start + width, size_t(length)), false, (flags & 4) != 0, 0);
    }
    if (const auto symbol = result.messages.find(17); symbol != result.messages.end()) {
        const auto local = remainder(address(symbol->second, address_width_));
        require(local.text(0, 4) == "HEAP" && local.le(4, 1) == 0, "Unsupported HDF5 local heap");
        const auto heap = absolute(address(local, 8 + 2 * size_width_), local.le(8, size_width_));
        symbol_tree(result, address(symbol->second, 0), heap, 0);
    }
    const auto links = result.messages.equal_range(6);
    for (auto link = links.first; link != links.second; ++link) inline_link(result, link->second);
    return objects_.emplace(offset, std::move(result)).first->second;
}

uint64_t Hdf5File::lookup(const std::string& path) {
    auto offset = root_;
    std::string_view parts = path;
    while (!parts.empty()) {
        if (parts.front() == '/') { parts.remove_prefix(1); continue; }
        const auto slash = parts.find('/');
        const auto name = std::string(parts.substr(0, slash));
        const auto& entries = object(offset).children;
        const auto found = entries.find(name);
        if (found == entries.end()) return UINT64_MAX;
        offset = found->second;
        parts.remove_prefix(slash == parts.npos ? parts.size() : slash + 1);
    }
    return offset;
}

bool Hdf5File::contains(const std::string& path) { return lookup(path) != UINT64_MAX; }

std::vector<std::string> Hdf5File::children(const std::string& path) {
    const auto offset = lookup(path);
    require(offset != UINT64_MAX, "Unknown HDF5 group " + path);
    std::vector<std::string> names;
    for (const auto& item : object(offset).children) names.push_back(item.first);
    return names;
}

std::string Hdf5File::heap_string(uint64_t offset, uint32_t index, uint64_t size) const {
    const auto start = remainder(offset);
    require(start.text(0, 4) == "GCOL" && start.le(4, 1) == 1, "Invalid HDF5 global heap");
    const auto heap = absolute(offset, start.le(8, size_width_));
    size_t pos = 8 + size_width_;
    while (heap.size - pos >= 8 + size_width_) {
        const auto entry = heap.le(pos, 2);
        if (!entry) break;
        const auto count = heap.le(pos + 8, size_width_);
        pos += 8 + size_width_;
        require(count <= heap.size - pos, "Invalid HDF5 heap object length");
        if (entry == index) {
            require(size <= count, "Invalid HDF5 string length");
            return heap.text(pos, size_t(size));
        }
        const auto aligned = align8(size_t(count));
        require(aligned <= heap.size - pos, "Invalid HDF5 heap padding");
        pos += aligned;
    }
    throw std::invalid_argument("HDF5 string heap object was not found");
}

std::string Hdf5File::attribute_text(const std::string& path, const std::string& name, bool required) {
    const auto offset = lookup(path);
    require(offset != UINT64_MAX, "Unknown HDF5 object " + path);
    const auto range = object(offset).messages.equal_range(12);
    for (auto item = range.first; item != range.second; ++item) {
        const auto bytes = item->second;
        const auto version = bytes.le(0, 1);
        require(version >= 1 && version <= 3, "Unsupported HDF5 attribute version");
        const auto name_size = size_t(bytes.le(2, 2)), type_size = size_t(bytes.le(4, 2)), shape_size = size_t(bytes.le(6, 2));
        const size_t start = version == 3 ? 9 : 8;
        if (cstring(bytes.sub(start, name_size)) != name) continue;
        const auto padded = [version](size_t count) { return version == 1 ? align8(count) : count; };
        const auto type_pos = start + padded(name_size), shape_pos = type_pos + padded(type_size);
        require(!(bytes.le(1, 1) & 3) && shape_element_count(shape(bytes.sub(shape_pos, shape_size), size_width_)) == 1,
                "HDF5 string attribute must have one unshared element");
        const auto value = shape_pos + padded(shape_size);
        const auto type = bytes.sub(type_pos, type_size);
        const auto family = type.le(0, 1) & 15;
        if (family == 3) return cstring(bytes.sub(value, size_t(type.le(4, 4))));
        require(family == 9 && (type.le(1, 1) & 15) == 1, "HDF5 attribute is not a string");
        return heap_string(address(bytes, value + 4), uint32_t(bytes.le(value + 4 + address_width_, 4)), bytes.le(value, 4));
    }
    if (required) throw std::invalid_argument("Missing HDF5 attribute " + name + " on " + path);
    return {};
}

Tensor Hdf5File::tensor(const std::string& path, Backend backend) {
    const auto offset = lookup(path);
    require(offset != UINT64_MAX, "Missing HDF5 dataset " + path);
    const auto& dataset = object(offset);
    const auto& messages = dataset.messages;
    auto required = [&](uint16_t id) {
        const auto found = messages.find(id);
        require(found != messages.end(), "Missing HDF5 dataset descriptor for " + path);
        return found->second;
    };
    const auto dimensions = shape(required(1), size_width_);
    const auto type = required(3), layout = required(8);
    const auto dtype = data_type(type);
    const auto count = shape_element_count(dimensions), length = dtype_storage_bytes(dtype, count);
    const auto version = layout.le(0, 1), storage = layout.le(version <= 2 ? 2 : 1, 1);
    require(version >= 1 && version <= 4 && storage <= 2, "Unsupported HDF5 dataset layout for " + path);
    Bytes bytes;
    std::vector<uint8_t> data;
    if (length) {
        if (storage == 2) {
            data = chunked(dataset, layout, dimensions, dtype_info(dtype).bits / 8);
        } else if (storage == 1) {
            const size_t start = version <= 2 ? 8 : 2;
            if (version >= 3) require(layout.le(start + address_width_, size_width_) == length, "Incorrect HDF5 dataset length");
            bytes = absolute(address(layout, start), length);
        } else {
            const auto size_pos = version <= 2 ? 8 + 4 * size_t(layout.le(1, 1)) : 2;
            const size_t width = version <= 2 ? 4 : 2;
            require(layout.le(size_pos, width) == length, "Incorrect HDF5 inline dataset length");
            bytes = layout.sub(size_pos + width, length);
        }
    }
    if (bytes.size) data.assign(bytes.data, bytes.data + bytes.size);
    if (type.le(1, 1) & 1) {
        const auto width = dtype_info(dtype).bits / 8;
        for (size_t i = 0; i < data.size(); i += width) std::reverse(data.begin() + i, data.begin() + i + width);
    }
    return Tensor::from_raw(dimensions, dtype, data, backend);
}

std::vector<uint8_t> Hdf5File::chunked(const Object& object, Bytes layout, const Shape& dimensions, size_t width) const {
    require(layout.le(0, 1) == 3 && layout.le(2, 1) == dimensions.size() + 1 && !dimensions.empty(),
            "Unsupported HDF5 chunk index layout");
    const auto index = address(layout, 3);
    require(defined(index), "HDF5 tensor has no allocated chunks");
    const int rank = int(dimensions.size());
    Shape chunk;
    size_t expected = 1;
    for (int axis = 0; axis < rank; ++axis) {
        const auto extent = layout.le(3 + address_width_ + 4 * axis, 4);
        require(extent > 0 && extent <= uint64_t(INT32_MAX), "Invalid HDF5 chunk extent");
        chunk.push_back(int(extent));
        require(dimensions[axis] > 0, "Invalid empty HDF5 chunked dataset");
        expected *= (dimensions[axis] - 1) / size_t(extent) + 1;
    }
    require(layout.le(3 + address_width_ + 4 * rank, 4) == width, "Invalid HDF5 chunk element width");
    const auto chunk_elements = shape_element_count(chunk);
    require(chunk_elements <= std::numeric_limits<size_t>::max() / width, "HDF5 chunk size overflow");
    const auto chunk_bytes = chunk_elements * width;
    const auto count = shape_element_count(dimensions);
    std::vector<uint8_t> out(count * width);
    std::set<Shape> positions;
    const auto pipeline = object.messages.find(11);
    const auto stages = pipeline == object.messages.end() ? std::vector<Filter>{} : filters(pipeline->second);
    auto apply = [&](Bytes key, uint64_t location) {
        const auto size = key.le(0, 4), mask = key.le(4, 4);
        Shape origin;
        for (int axis = 0; axis < rank; ++axis) {
            const auto value = key.le(8 + 8 * axis, 8);
            require(value < uint64_t(dimensions[axis]) && value % chunk[axis] == 0, "Invalid HDF5 chunk position");
            origin.push_back(int(value));
        }
        require(!key.le(8 + 8 * rank, 8) && positions.insert(origin).second, "Duplicate or invalid HDF5 chunk");
        const auto stored = absolute(location, size);
        std::vector<uint8_t> decoded(stored.data, stored.data + stored.size);
        for (size_t stage = stages.size(); stage > 0;) {
            --stage;
            if (mask & (uint64_t(1) << stage)) continue;
            const auto& filter = stages[stage];
            if (filter.id == 1) decoded = inflate_zlib(view(decoded), chunk_bytes);
            else if (filter.id == 2) {
                require(decoded.size() == chunk_bytes && filter.parameters.size() == 1 && filter.parameters[0] == width,
                        "Invalid HDF5 shuffle filter");
                std::vector<uint8_t> interleaved(chunk_bytes);
                for (size_t i = 0; i < chunk_elements; ++i)
                    for (size_t byte = 0; byte < width; ++byte) interleaved[i * width + byte] = decoded[byte * chunk_elements + i];
                decoded = std::move(interleaved);
            } else throw std::invalid_argument("Unsupported HDF5 dataset filter " + std::to_string(filter.id));
        }
        require(decoded.size() == chunk_bytes, "Incorrect decoded HDF5 chunk length");
        for (size_t element = 0; element < chunk_elements; ++element) {
            size_t index = element, target = 0, stride = 1;
            bool visible = true;
            for (int axis = rank - 1; axis >= 0; --axis) {
                const auto coordinate = size_t(origin[axis]) + index % size_t(chunk[axis]);
                if (coordinate >= size_t(dimensions[axis])) { visible = false; break; }
                target += coordinate * stride;
                index /= size_t(chunk[axis]);
                stride *= size_t(dimensions[axis]);
            }
            if (visible) std::copy_n(decoded.data() + element * width, width, out.data() + target * width);
        }
    };
    const size_t key_width = 8 + 8 * (size_t(rank) + 1);
    auto walk = [&](auto&& walk, uint64_t location, unsigned depth) -> void {
        require(depth < 32, "HDF5 chunk tree exceeds the importer limit");
        const auto tree = remainder(location);
        require(tree.text(0, 4) == "TREE" && tree.le(4, 1) == 1, "Invalid HDF5 chunk tree");
        const auto level = tree.le(5, 1), entries = tree.le(6, 2);
        size_t offset = 8 + 2 * address_width_;
        for (size_t i = 0; i < entries; ++i) {
            const auto key = tree.sub(offset, key_width);
            const auto value = address(tree, offset + key_width);
            if (level) {
                require(absolute(value, 8).le(5, 1) + 1 == level, "Invalid HDF5 chunk tree depth");
                walk(walk, value, depth + 1);
            } else apply(key, value);
            offset += key_width + address_width_;
        }
    };
    walk(walk, index, 0);
    require(positions.size() == expected, "HDF5 tensor contains unallocated chunks");
    return out;
}

}  // namespace elman::io
