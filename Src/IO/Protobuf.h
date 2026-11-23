#pragma once

#include "IO/BinaryInput.h"

namespace elman::io::protobuf {

inline uint64_t varint(Bytes bytes, size_t& offset) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift <= 63; shift += 7) {
        const uint64_t next = bytes.le(offset++, 1);
        require(shift != 63 || next <= 1, "Protobuf varint exceeds 64 bits");
        value |= (next & 127) << shift;
        if (!(next & 128)) return value;
    }
    throw std::invalid_argument("Unterminated protobuf varint");
}

struct Field {
    uint32_t number;
    unsigned wire;
    uint64_t value = 0;
    Bytes data;

    uint64_t integer() const {
        require(wire != 2, "Expected a protobuf numeric field");
        return value;
    }
    Bytes bytes() const {
        require(wire == 2, "Expected a protobuf bytes field");
        return data;
    }
};

class Message {
public:
    explicit Message(Bytes bytes = {}) {
        size_t offset = 0;
        while (offset < bytes.size) {
            const uint64_t tag = varint(bytes, offset);
            require((tag >> 3) > 0 && (tag >> 3) < (1u << 29), "Invalid protobuf tag");
            Field field{uint32_t(tag >> 3), unsigned(tag & 7)};
            switch (field.wire) {
                case 0: field.value = varint(bytes, offset); break;
                case 1: field.value = bytes.le(offset, 8); offset += 8; break;
                case 2: {
                    const uint64_t length = varint(bytes, offset);
                    require(length <= bytes.size - offset, "Truncated protobuf field");
                    field.data = bytes.sub(offset, size_t(length));
                    offset += size_t(length);
                    break;
                }
                case 5: field.value = bytes.le(offset, 4); offset += 4; break;
                default: throw std::invalid_argument("Unsupported protobuf wire type");
            }
            fields_.push_back(field);
        }
    }
    const Field* field(uint32_t number) const {
        for (auto it = fields_.rbegin(); it != fields_.rend(); ++it)
            if (it->number == number) return &*it;
        return nullptr;
    }
    Bytes bytes(uint32_t number) const {
        const Field* const value = field(number);
        return value ? value->bytes() : Bytes{};
    }
    std::string string(uint32_t number) const {
        const Bytes value = bytes(number);
        return value.text(0, value.size);
    }
    uint64_t integer(uint32_t number, uint64_t fallback = 0) const {
        const Field* const value = field(number);
        return value ? value->integer() : fallback;
    }
    std::vector<Bytes> messages(uint32_t number) const {
        std::vector<Bytes> values;
        for (const io::protobuf::Field& field : fields_) if (field.number == number) values.push_back(field.bytes());
        return values;
    }
    std::vector<uint64_t> numbers(uint32_t number, size_t fixed_width = 0) const {
        std::vector<uint64_t> values;
        for (const io::protobuf::Field& field : fields_) {
            if (field.number != number) continue;
            if (field.wire != 2) values.push_back(field.value);
            else {
                size_t offset = 0;
                while (offset < field.data.size) {
                    if (fixed_width) {
                        values.push_back(field.data.le(offset, fixed_width));
                        offset += fixed_width;
                    } else values.push_back(varint(field.data, offset));
                }
            }
        }
        return values;
    }

private:
    std::vector<Field> fields_;
};

}  // namespace elman::io::protobuf
