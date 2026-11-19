#include "IO/TensorFlowBundle.h"

#include <array>
#include <iomanip>
#include <sstream>

namespace elman::io::tensorflow {
namespace {

uint32_t masked_crc(Bytes bytes) {
    static const auto table = [] {
        std::array<uint32_t, 256> lookup{};
        for (uint32_t i = 0; i < lookup.size(); ++i) {
            uint32_t crc = i;
            for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1) ? 0x82f63b78U : 0);
            lookup[i] = crc;
        }
        return lookup;
    }();
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < bytes.size; ++i) crc = table[(crc ^ bytes.data[i]) & 255] ^ (crc >> 8);
    crc ^= UINT32_MAX;
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8U;
}

std::vector<uint8_t> snappy(Bytes data) {
    size_t offset = 0;
    const auto length = protobuf::varint(data, offset);
    require(length <= std::numeric_limits<size_t>::max() &&
            (length / 64 + (length % 64 != 0)) <= data.size, "Invalid Snappy block length");
    std::vector<uint8_t> decoded;
    decoded.reserve(size_t(length));
    while (offset < data.size) {
        const auto tag = data.le(offset++, 1), kind = tag & 3;
        if (kind == 0) {
            auto size = tag >> 2;
            if (size >= 60) {
                const auto width = size_t(size - 59);
                size = data.le(offset, width);
                offset += width;
            }
            ++size;
            require(size <= length - decoded.size(), "Invalid Snappy literal length");
            const auto literal = data.sub(offset, size_t(size));
            decoded.insert(decoded.end(), literal.data, literal.data + literal.size);
            offset += literal.size;
        } else {
            const auto width = size_t(kind == 3 ? 4 : kind);
            uint64_t distance = data.le(offset, width);
            offset += width;
            if (kind == 1) distance |= (tag & 0xe0) << 3;
            const auto count = kind == 1 ? ((tag >> 2) & 7) + 4 : (tag >> 2) + 1;
            require(distance && distance <= decoded.size() && count <= length - decoded.size(), "Invalid Snappy copy");
            for (uint64_t i = 0; i < count; ++i) decoded.push_back(decoded[decoded.size() - size_t(distance)]);
        }
    }
    require(decoded.size() == length, "Incomplete Snappy block");
    return decoded;
}

std::vector<uint8_t> read_block(BinaryInput& file, Bytes handle, size_t& position) {
    const auto start = protobuf::varint(handle, position);
    const auto size = protobuf::varint(handle, position);
    require(start <= file.size() && file.size() - start >= 5 && size <= file.size() - start - 5,
            "Invalid TensorFlow checkpoint table block");
    file.seek(start);
    auto data = file.bytes(size + 1);
    const auto checksum = file.le(4);
    require(masked_crc(view(data)) == checksum, "TensorFlow index block checksum mismatch");
    const auto codec = data.back();
    data.pop_back();
    if (codec == 1) return snappy(view(data));
    require(codec == 0, "Unsupported TensorFlow table compression");
    return data;
}

template<class ReadEntry>
void block_entries(Bytes bytes, ReadEntry accept) {
    require(bytes.size >= 4, "Truncated TensorFlow index block");
    const auto restart_count = bytes.le(bytes.size - 4, 4);
    require(restart_count && restart_count <= bytes.size / 4 - 1, "Invalid TensorFlow index restart count");
    const auto records = bytes.sub(0, bytes.size - 4 * (size_t(restart_count) + 1));
    std::string key;
    size_t offset = 0;
    while (offset < records.size) {
        const auto shared = protobuf::varint(records, offset);
        const auto suffix = protobuf::varint(records, offset);
        const auto length = protobuf::varint(records, offset);
        require(shared <= key.size() && suffix <= records.size - offset && length <= records.size - offset - suffix,
                "Invalid TensorFlow index entry");
        key.resize(size_t(shared));
        key += records.text(offset, size_t(suffix));
        offset += size_t(suffix);
        accept(key, records.sub(offset, size_t(length)));
        offset += size_t(length);
    }
}

}  // namespace

Bundle::Bundle(const std::filesystem::path& prefix) : prefix_(prefix) {
    auto index = prefix;
    index += ".index";
    BinaryInput file(index);
    require(file.size() >= 48, "Truncated TensorFlow checkpoint index");
    file.seek(file.size() - 48);
    const auto footer_data = file.bytes(48);
    const auto footer = view(footer_data);
    require(footer.le(40, 8) == 0xdb4775248b80fb57ULL, "Invalid TensorFlow checkpoint table signature");
    const auto handles = footer.sub(0, 40);
    size_t offset = 0;
    protobuf::varint(handles, offset);
    protobuf::varint(handles, offset);
    const auto catalog = read_block(file, handles, offset);
    block_entries(view(catalog), [&](const std::string&, Bytes handle) {
        size_t position = 0;
        const auto block = read_block(file, handle, position);
        require(position == handle.size, "Invalid TensorFlow checkpoint block handle");
        block_entries(view(block), [&](const std::string& key, Bytes bytes) {
            const protobuf::Message info(bytes);
            if (key.empty()) {
                require(!shards_ && info.integer(2) == 0, "Unsupported TensorFlow checkpoint header");
                shards_ = info.integer(1);
                require(shards_ > 0 && shards_ <= 99999, "Invalid TensorFlow checkpoint shard count");
                return;
            }
            const auto type = info.integer(1);
            if (type == 7 || type == 20 || type == 21 || key == "_CHECKPOINTABLE_OBJECT_GRAPH") return;
            require(shards_ && info.integer(3) < shards_ && !info.field(7), "Unsupported TensorFlow checkpoint tensor storage");
            Entry entry{dtype(type), dimensions(protobuf::Message(info.bytes(2))),
                        info.integer(3), info.integer(4), info.integer(5), uint32_t(info.integer(6))};
            require(entries_.emplace(key, std::move(entry)).second, "Duplicate TensorFlow checkpoint tensor " + key);
        });
    });
    require(shards_ != 0, "Missing TensorFlow checkpoint header");
}

Tensor Bundle::tensor(const std::string& name, Backend backend) const {
    const auto found = entries_.find(name);
    require(found != entries_.end(), "Missing TensorFlow checkpoint tensor " + name);
    const auto& e = found->second;
    std::ostringstream suffix;
    suffix << ".data-" << std::setfill('0') << std::setw(5) << e.shard << "-of-" << std::setw(5) << shards_;
    auto path = prefix_;
    path += suffix.str();
    BinaryInput file(path);
    file.seek(e.offset);
    require(e.size == dtype_storage_bytes(e.type, shape_element_count(e.shape)), "Invalid TensorFlow checkpoint tensor length");
    const auto data = file.bytes(e.size);
    require(masked_crc(view(data)) == e.checksum, "TensorFlow checkpoint tensor checksum mismatch for " + name);
    return Tensor::from_raw(e.shape, e.type, data, backend);
}

}  // namespace elman::io::tensorflow
