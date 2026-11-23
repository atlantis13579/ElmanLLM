#include "IO/ZipArchive.h"
#include "IO/Deflate.h"

#include <algorithm>
#include <array>

namespace elman::io {
namespace {

class BitStream {
public:
    explicit BitStream(Bytes bytes) : bytes_(bytes) {}
    unsigned read(unsigned count) {
        unsigned value = 0;
        for (unsigned i = 0; i < count; ++i, ++bit_) {
            require(bit_ / 8 < bytes_.size, "Truncated DEFLATE block");
            value |= ((bytes_.data[bit_ / 8] >> (bit_ % 8)) & 1u) << i;
        }
        return value;
    }
    void align() { bit_ += (8 - bit_ % 8) % 8; }
private:
    Bytes bytes_;
    size_t bit_ = 0;
};

class Huffman {
public:
    explicit Huffman(const std::vector<unsigned>& lengths) {
        std::array<unsigned, 16> counts{}, next{};
        for (unsigned int length : lengths) {
            require(length <= 15, "Invalid DEFLATE Huffman length");
            if (length) ++counts[length];
        }
        for (size_t length = 1; length < counts.size(); ++length) {
            next[length] = (next[length - 1] + counts[length - 1]) * 2;
            require(next[length] + counts[length] <= (1u << length), "Oversubscribed DEFLATE Huffman tree");
        }
        nodes_.push_back({});
        for (size_t symbol = 0; symbol < lengths.size(); ++symbol) {
            int length = static_cast<int>(lengths[symbol]);
            if (!length) continue;
            const unsigned code = next[length]++;
            int node = 0;
            while (length--) {
                const unsigned int branch = (code >> length) & 1;
                int child = nodes_[node].child[branch];
                if (child < 0) {
                    child = static_cast<int>(nodes_.size());
                    nodes_[node].child[branch] = child;
                    nodes_.push_back({});
                }
                node = child;
            }
            nodes_[node].symbol = static_cast<int>(symbol);
        }
    }
    unsigned decode(BitStream& stream) const {
        int node = 0;
        while (nodes_[node].symbol < 0) {
            node = nodes_[node].child[stream.read(1)];
            require(node >= 0, "Invalid DEFLATE symbol");
        }
        return static_cast<unsigned>(nodes_[node].symbol);
    }
private:
    struct Node { int child[2] = {-1, -1}; int symbol = -1; };
    std::vector<Node> nodes_;
};

std::pair<Huffman, Huffman> tables(BitStream& stream, unsigned type) {
    if (type == 1) {
        std::vector<unsigned> code(288, 8);
        std::fill(code.begin() + 144, code.begin() + 256, 9);
        std::fill(code.begin() + 256, code.begin() + 280, 7);
        return {Huffman(code), Huffman(std::vector<unsigned>(32, 5))};
    }
    require(type == 2, "Reserved DEFLATE block type");
    const unsigned literals = stream.read(5) + 257, distances = stream.read(5) + 1, count = stream.read(4) + 4;
    require(literals <= 286, "Invalid DEFLATE literal alphabet");
    const unsigned order[] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
    std::vector<unsigned> codes(19), lengths;
    for (unsigned i = 0; i < count; ++i) codes[order[i]] = stream.read(3);
    const Huffman code_lengths(codes);
    while (lengths.size() < literals + distances) {
        const unsigned int symbol = code_lengths.decode(stream);
        if (symbol < 16) {
            lengths.push_back(symbol);
            continue;
        }
        require(symbol != 16 || !lengths.empty(), "DEFLATE repeat has no previous code length");
        const unsigned int times = symbol == 16 ? stream.read(2) + 3 : symbol == 17 ? stream.read(3) + 3 : stream.read(7) + 11;
        require(times <= literals + distances - lengths.size(), "DEFLATE code repeat exceeds alphabet");
        lengths.insert(lengths.end(), times, symbol == 16 ? lengths.back() : 0);
    }
    require(lengths[256] != 0, "DEFLATE block has no terminator");
    return {Huffman({lengths.begin(), lengths.begin() + literals}), Huffman({lengths.begin() + literals, lengths.end()})};
}

std::vector<uint8_t> inflate(Bytes bytes, size_t size) {
    static constexpr unsigned length_base[] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static constexpr unsigned distance_base[] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    BitStream stream(bytes);
    std::vector<uint8_t> out;
    out.reserve(std::min<size_t>(size, 1024 * 1024));
    bool last;
    do {
        last = stream.read(1) != 0;
        const unsigned int type = stream.read(2);
        if (type == 0) {
            stream.align();
            const unsigned int count = stream.read(16);
            require(stream.read(16) == (count ^ 65535u) && count <= size - out.size(), "Invalid stored DEFLATE block");
            for (unsigned i = 0; i < count; ++i) out.push_back(static_cast<uint8_t>(stream.read(8)));
            continue;
        }
        const std::pair<Huffman, Huffman> codes = tables(stream, type);
        const Huffman& literal = codes.first;
        const Huffman& distances = codes.second;
        while (true) {
            const unsigned int symbol = literal.decode(stream);
            if (symbol == 256) break;
            if (symbol < 256) {
                require(out.size() < size, "DEFLATE output exceeds ZIP entry size");
                out.push_back(static_cast<uint8_t>(symbol));
                continue;
            }
            const unsigned code = symbol - 257;
            require(code < 29, "Invalid DEFLATE length symbol");
            const unsigned length = length_base[code] + stream.read(code < 8 || code == 28 ? 0 : code / 4 - 1);
            const unsigned int distance_code = distances.decode(stream);
            require(distance_code < 30, "Invalid DEFLATE distance symbol");
            const unsigned distance = distance_base[distance_code] + stream.read(distance_code < 4 ? 0 : distance_code / 2 - 1);
            require(distance <= out.size() && length <= size - out.size(), "Invalid DEFLATE back-reference");
            for (unsigned i = 0; i < length; ++i) out.push_back(out[out.size() - distance]);
        }
    } while (!last);
    require(out.size() == size, "ZIP entry decompressed to the wrong size");
    return out;
}

uint32_t crc32(Bytes bytes) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            uint32_t n = static_cast<uint32_t>(i);
            for (int bit = 0; bit < 8; ++bit) n = (n >> 1) ^ ((n & 1) ? 0xedb88320u : 0);
            values[i] = n;
        }
        return values;
    }();
    uint32_t code = 0xffffffffu;
    for (size_t i = 0; i < bytes.size; ++i) code = table[(code ^ bytes.data[i]) & 255] ^ (code >> 8);
    return ~code;
}

}  // namespace

std::vector<uint8_t> inflate_zlib(Bytes bytes, size_t size) {
    const uint64_t method = bytes.le(0, 1), flags = bytes.le(1, 1);
    require(bytes.size >= 6 && (method & 15) == 8 && (method >> 4) <= 7 &&
            ((method << 8) | flags) % 31 == 0 && !(flags & 32), "Unsupported zlib stream header");
    const std::vector<uint8_t> result = inflate(bytes.sub(2, bytes.size - 6), size);
    uint32_t first = 1, second = 0;
    for (const unsigned char byte : result) {
        first = (first + byte) % 65521;
        second = (second + first) % 65521;
    }
    const Bytes trailer = bytes.sub(bytes.size - 4, 4);
    const uint64_t expected = (trailer.le(0, 1) << 24) | (trailer.le(1, 1) << 16) |
                          (trailer.le(2, 1) << 8) | trailer.le(3, 1);
    require(((second << 16) | first) == expected, "zlib Adler-32 checksum mismatch");
    return result;
}

ZipArchive::ZipArchive(const std::filesystem::path& path) : input_(path) {
    constexpr uint64_t limit = 65535 + 22;
    const uint64_t start = input_.size() - std::min(input_.size(), limit);
    input_.seek(start);
    const std::vector<uint8_t> tail = input_.bytes(input_.remaining());
    const Bytes data = view(tail);
    require(data.size >= 22, "Truncated ZIP archive");
    size_t end = data.size - 22;
    while (data.le(end, 4) != 0x06054b50u || end + 22 + data.le(end + 20, 2) != data.size) {
        require(end > 0, "ZIP directory end record missing");
        --end;
    }
    require(data.le(end + 4, 4) == 0 && data.le(end + 8, 2) == data.le(end + 10, 2), "Multi-disk ZIP is unsupported");
    uint64_t count = data.le(end + 10, 2), bytes = data.le(end + 12, 4);
    directory_ = data.le(end + 16, 4);
    uint64_t directory_end = start + end;
    if (count == 65535 || bytes == 0xffffffffu || directory_ == 0xffffffffu ||
        (end >= 20 && data.le(end - 20, 4) == 0x07064b50u)) {
        require(start + end >= 20, "Missing ZIP64 locator");
        input_.seek(start + end - 20);
        require(input_.le(4) == 0x07064b50u && input_.le(4) == 0, "Invalid ZIP64 locator");
        const uint64_t record = input_.le(8);
        require(input_.le(4) == 1 && record <= start + end - 20 && start + end - 20 - record >= 56, "Invalid ZIP64 directory end");
        directory_end = record;
        input_.seek(record);
        require(input_.le(4) == 0x06064b50u && input_.le(8) >= 44, "Invalid ZIP64 directory end");
        input_.skip(4);
        require(input_.le(8) == 0, "Multi-disk ZIP64 is unsupported");
        const uint64_t disk_count = input_.le(8);
        count = input_.le(8);
        require(disk_count == count, "Multi-disk ZIP64 is unsupported");
        bytes = input_.le(8);
        directory_ = input_.le(8);
    }
    require(directory_ <= directory_end && bytes <= directory_end - directory_ && count <= bytes / 46, "Invalid ZIP directory range");
    const uint64_t end_offset = directory_ + bytes;
    input_.seek(directory_);
    for (uint64_t i = 0; i < count; ++i) {
        const std::vector<uint8_t> header = input_.bytes(46);
        const Bytes h = view(header);
        const uint64_t flags = h.le(8, 2), method = h.le(10, 2);
        require(h.le(0, 4) == 0x02014b50u && !(flags & 1) && (method == 0 || method == 8) && h.le(34, 2) == 0,
                "Unsupported ZIP entry header");
        const std::string name = input_.text(h.le(28, 2));
        require(!name.empty() && name.find('\0') == std::string::npos, "Invalid ZIP entry name");
        const std::vector<uint8_t> extra = input_.bytes(h.le(30, 2));
        input_.skip(h.le(32, 2));
        require(input_.position() <= end_offset, "ZIP entry exceeds central directory");
        Entry entry{h.le(42, 4), h.le(20, 4), h.le(24, 4), static_cast<uint32_t>(h.le(16, 4)), static_cast<uint16_t>(method)};
        if (entry.offset == 0xffffffffu || entry.compressed == 0xffffffffu || entry.size == 0xffffffffu) {
            Bytes fields = view(extra);
            while (fields.size >= 4 && fields.le(0, 2) != 1) {
                const size_t length = 4 + static_cast<size_t>(fields.le(2, 2));
                require(length <= fields.size, "Truncated ZIP extra field");
                fields = fields.sub(length, fields.size - length);
            }
            require(fields.size >= 4, "ZIP64 extra field missing");
            fields = fields.sub(4, static_cast<size_t>(fields.le(2, 2)));
            size_t at = 0;
            for (uint64_t* value : {&entry.size, &entry.compressed, &entry.offset}) {
                if (*value == 0xffffffffu) {
                    *value = fields.le(at, 8);
                    at += 8;
                }
            }
        }
        require(entry.offset < directory_ && entry.compressed <= directory_ - entry.offset &&
                entries_.emplace(name, entry).second, "Invalid or duplicate ZIP entry");
    }
    require(input_.position() == end_offset, "ZIP directory length differs from its entries");
}

std::vector<uint8_t> ZipArchive::read(const std::string& name) {
    const auto found = entries_.find(name);
    require(found != entries_.end(), "Missing ZIP entry: " + name);
    const io::ZipArchive::Entry& entry = found->second;
    input_.seek(entry.offset);
    const std::vector<uint8_t> header = input_.bytes(30);
    const Bytes h = view(header);
    require(h.le(0, 4) == 0x04034b50u && h.le(8, 2) == entry.method && !(h.le(6, 2) & 1) &&
            input_.text(h.le(26, 2)) == name, "ZIP local header differs from the catalog");
    input_.skip(h.le(28, 2));
    require(input_.position() <= directory_ && entry.compressed <= directory_ - input_.position() &&
            entry.size <= std::numeric_limits<size_t>::max(), "Invalid ZIP payload range");
    if (entry.method == 8) {
        // A maximum-length back reference takes at least two compressed bits.
        constexpr uint64_t max_expansion = 4 * 258;
        require(entry.compressed >= entry.size / max_expansion + (entry.size % max_expansion != 0),
                "DEFLATE entry exceeds the maximum expansion ratio");
    }
    std::vector<uint8_t> data = input_.bytes(entry.compressed);
    if (entry.method == 8) data = inflate(view(data), static_cast<size_t>(entry.size));
    require(data.size() == entry.size && crc32(view(data)) == entry.checksum, "ZIP entry CRC or size mismatch: " + name);
    return data;
}

}  // namespace elman::io
