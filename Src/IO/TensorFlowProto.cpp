#include "IO/TensorFlowProto.h"

#include <algorithm>

namespace elman::io::tensorflow {

Tensor constant(const protobuf::Message& proto, Backend backend) {
    const DType type = dtype(proto.integer(1));
    const Shape shape = dimensions(protobuf::Message(proto.bytes(2)));
    const size_t count = shape_element_count(shape);
    const size_t size = dtype_storage_bytes(type, count);
    const Bytes content = proto.bytes(4);
    if (content.size) {
        require(content.size == size, "TensorFlow tensor content has the wrong size");
        return Tensor::from_raw(shape, type, {content.data, content.data + content.size}, backend);
    }
    uint32_t field = 7;
    size_t fixed = 0, components = 1;
    switch (type) {
        case DType::Float32: field = 5; fixed = 4; break;
        case DType::Float64: field = 6; fixed = 8; break;
        case DType::Float16: case DType::BFloat16: field = 13; break;
        case DType::Int64: field = 10; break;
        case DType::Bool: field = 11; break;
        case DType::UInt32: field = 16; break;
        case DType::UInt64: field = 17; break;
        case DType::Complex64: field = 9; fixed = 4; components = 2; break;
        case DType::Complex128: field = 12; fixed = 8; components = 2; break;
        default: break;
    }
    const std::vector<uint64_t> values = proto.numbers(field, fixed);
    require(values.size() % components == 0 && values.size() / components <= count, "Invalid TensorFlow tensor values");
    std::vector<uint8_t> data(size);
    const size_t width = dtype_info(type).bits / (8 * components);
    if (!values.empty()) {
        for (size_t i = 0; i < count; ++i) {
            const uint64_t source = std::min(i, values.size() / components - 1);
            for (size_t component = 0; component < components; ++component) {
                const uint64_t bits = values[source * components + component];
                const size_t start = (i * components + component) * width;
                for (size_t byte = 0; byte < width; ++byte) data[start + byte] = uint8_t(bits >> (byte * 8));
            }
        }
    }
    return Tensor::from_raw(shape, type, data, backend);
}

}  // namespace elman::io::tensorflow
