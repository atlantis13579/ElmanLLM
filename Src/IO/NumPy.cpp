#include "IO/NumPy.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <string_view>
#include <fstream>
#include <sstream>
#include "IO/AtomicFile.h"

#include "Core/TensorDispatch.h"
#include "IO/ZipArchive.h"

namespace elman::io {
void save_npy(const std::filesystem::path& path, const Tensor& tensor) {
    std::string descriptor;
    switch (tensor.dtype()) {
    case DType::Float32: descriptor = "<f4"; break;
    case DType::Float64: descriptor = "<f8"; break;
    case DType::Int32: descriptor = "<i4"; break;
    case DType::Int64: descriptor = "<i8"; break;
    case DType::UInt8: descriptor = "|u1"; break;
    default: throw std::invalid_argument("NPY export dtype is unsupported");
    }
    std::ostringstream shape;
    for (int dim : tensor.shape()) shape << dim << ", ";
    std::string header = "{'descr': '" + descriptor + "', 'fortran_order': False, 'shape': (" + shape.str() + "), }";
    header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
    header += '\n';
    if (header.size() > 65535) throw std::invalid_argument("NPY header is too large");
    const auto bytes = tensor.raw_data();
    write_file_atomically(path, [&](const std::filesystem::path& temporary) {
        std::ofstream stream(temporary, std::ios::binary);
        const char prefix[] = {char(0x93),'N','U','M','P','Y',1,0,char(header.size() & 255),char(header.size() >> 8)};
        stream.write(prefix, sizeof(prefix));
        stream.write(header.data(), std::streamsize(header.size()));
        stream.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        stream.close();
        if (!stream) throw std::runtime_error("NPY write failed: " + path.u8string());
    });
}

namespace {

struct Layout { std::string descriptor; Shape shape; bool fortran = false; };

class Header {
public:
    explicit Header(std::string_view text) : text_(text) {}
    Layout parse() {
        Layout layout;
        bool dtype = false, order = false, shape = false;
        expect('{');
        while (!take('}')) {
            const std::string key = quoted();
            expect(':');
            if (key == "descr" && !dtype) {
                layout.descriptor = quoted();
                dtype = true;
            } else if (key == "fortran_order" && !order) {
                whitespace();
                if (text_.substr(0, 4) == "True") { layout.fortran = true; text_.remove_prefix(4); }
                else if (text_.substr(0, 5) == "False") text_.remove_prefix(5);
                else throw std::invalid_argument("Invalid NumPy storage order");
                order = true;
            } else if (key == "shape" && !shape) {
                expect('(');
                if (!take(')')) {
                    do {
                        whitespace();
                        int dim = 0;
                        const std::from_chars_result result = std::from_chars(text_.data(), text_.data() + text_.size(), dim);
                        require(result.ec == std::errc{} && dim >= 0 && layout.shape.size() < 8, "Invalid NumPy dimension or rank");
                        text_.remove_prefix(static_cast<size_t>(result.ptr - text_.data()));
                        layout.shape.push_back(dim);
                        if (!take(',')) { expect(')'); break; }
                    } while (!take(')'));
                }
                shape = true;
            } else {
                throw std::invalid_argument("Unknown or duplicate NumPy header field: " + key);
            }
            if (!take(',')) { expect('}'); break; }
        }
        whitespace();
        require(dtype && order && shape && text_.empty(), "Incomplete or malformed NumPy header");
        return layout;
    }

private:
    void whitespace() {
        while (!text_.empty() && std::isspace(static_cast<unsigned char>(text_.front()))) text_.remove_prefix(1);
    }
    bool take(char value) {
        whitespace();
        if (text_.empty() || text_.front() != value) return false;
        text_.remove_prefix(1);
        return true;
    }
    void expect(char value) { require(take(value), "Malformed NumPy array header"); }
    std::string quoted() {
        whitespace();
        require(!text_.empty() && (text_.front() == '\'' || text_.front() == '"'), "Expected a NumPy header string");
        const char delimiter = text_.front();
        text_.remove_prefix(1);
        const size_t end = text_.find(delimiter);
        require(end != std::string_view::npos, "Unterminated NumPy header string");
        std::string value(text_.substr(0, end));
        text_.remove_prefix(end + 1);
        return value;
    }
    std::string_view text_;
};

DType scalar_dtype(std::string_view value) {
    static const std::map<std::string_view, DType> types{
        {"b1", DType::Bool}, {"?", DType::Bool}, {"f2", DType::Float16}, {"f4", DType::Float32}, {"f8", DType::Float64},
        {"i1", DType::Int8}, {"i2", DType::Int16}, {"i4", DType::Int32}, {"i8", DType::Int64},
        {"u1", DType::UInt8}, {"u2", DType::UInt16}, {"u4", DType::UInt32}, {"u8", DType::UInt64},
        {"c8", DType::Complex64}, {"c16", DType::Complex128},
    };
    const auto type = types.find(value);
    if (type == types.end()) throw std::invalid_argument("Unsupported NumPy dtype: " + std::string(value));
    return type->second;
}

}  // namespace

Tensor npy_tensor(Bytes bytes, Backend backend) {
    require(bytes.size >= 10 && bytes.text(0, 6) == "\x93NUMPY", "Invalid NumPy array signature");
    const uint64_t major = bytes.le(6, 1);
    require(major >= 1 && major <= 3 && bytes.le(7, 1) == 0, "Unsupported NumPy array version");
    const size_t width = major == 1 ? 2 : 4;
    const size_t length = static_cast<size_t>(bytes.le(8, width));
    require(length <= bytes.size - (8 + width), "Truncated NumPy header");
    const std::string header = bytes.text(8 + width, length);
    const Layout layout = Header(header).parse();
    require(layout.descriptor.size() >= 2 && std::string_view("<>=|").find(layout.descriptor.front()) != std::string_view::npos,
            "Unsupported NumPy dtype byte order");
    const DType dtype = scalar_dtype(std::string_view(layout.descriptor).substr(1));
    const size_t item_bytes = dtype_info(dtype).bits / 8;
    require(layout.descriptor.front() != '|' || item_bytes == 1, "Invalid NumPy dtype byte order");
    const size_t count = shape_element_count(layout.shape);
    const size_t byte_count = dtype_storage_bytes(dtype, count);
    const size_t start = 8 + width + length;
    require(bytes.size - start == byte_count, "NumPy array payload size differs from its shape");
    std::vector<uint8_t> result(byte_count);
    const Bytes source = bytes.sub(start, byte_count);
    if (!layout.fortran || layout.shape.size() <= 1) {
        if (byte_count) std::copy_n(source.data, byte_count, result.data());
    } else if (count) {
        std::vector<size_t> strides(layout.shape.size(), 1);
        for (size_t axis = 1; axis < layout.shape.size(); ++axis) strides[axis] = strides[axis - 1] * layout.shape[axis - 1];
        for (size_t i = 0; i < count; ++i) {
            size_t index = 0, remaining = i;
            for (size_t axis = layout.shape.size(); axis > 0;) {
                --axis;
                const size_t dim = static_cast<size_t>(layout.shape[axis]);
                index += (remaining % dim) * strides[axis];
                remaining /= dim;
            }
            std::copy_n(source.data + index * item_bytes, item_bytes, result.data() + i * item_bytes);
        }
    }
    if (layout.descriptor.front() == '>') {
        const size_t component = is_complex(dtype) ? item_bytes / 2 : item_bytes;
        for (size_t i = 0; i < result.size(); i += component)
            std::reverse(result.begin() + i, result.begin() + i + component);
    }
    return Tensor::from_raw(layout.shape, dtype, result, backend);
}

TensorArchive numpy_archive(const std::filesystem::path& path, Backend backend, const std::string& single_name) {
    if (model_file_extension(path) == ".npy") {
        BinaryInput input(path);
        return {{single_name, npy_tensor(view(input.bytes(input.size())), backend)}};
    }
    TensorArchive tensors;
    ZipArchive archive(path);
    for (const std::pair<const std::string, io::ZipArchive::Entry>& entry : archive.entries()) {
        const std::string& name = entry.first;
        if (name.back() == '/' && entry.second.size == 0) continue;
        require(name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0, "NPZ entry is not a named NumPy array: " + name);
        require(tensors.emplace(name.substr(0, name.size() - 4), npy_tensor(view(archive.read(name)), backend)).second,
                "Duplicate NPZ tensor: " + name);
    }
    require(!tensors.empty(), "NPZ archive has no arrays");
    return tensors;
}

}  // namespace elman::io
