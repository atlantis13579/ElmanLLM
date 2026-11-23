#pragma once

#include "Core/TensorDispatch.h"
#include "IO/Protobuf.h"

namespace elman::io::tensorflow {

inline DType dtype(uint64_t id) {
    switch (id) {
        case 1: return DType::Float32;
        case 2: return DType::Float64;
        case 3: return DType::Int32;
        case 4: return DType::UInt8;
        case 5: return DType::Int16;
        case 6: return DType::Int8;
        case 8: return DType::Complex64;
        case 9: return DType::Int64;
        case 10: return DType::Bool;
        case 14: return DType::BFloat16;
        case 17: return DType::UInt16;
        case 18: return DType::Complex128;
        case 19: return DType::Float16;
        case 22: return DType::UInt32;
        case 23: return DType::UInt64;
        default: throw std::invalid_argument("Unsupported TensorFlow dtype " + std::to_string(id));
    }
}

inline Shape dimensions(const protobuf::Message& descriptor, bool signature = false) {
    Shape shape;
    for (io::Bytes bytes : descriptor.messages(2)) {
        const uint64_t dimension = protobuf::Message(bytes).integer(1);
        require(dimension <= uint64_t(std::numeric_limits<int>::max()) || (signature && dimension == UINT64_MAX),
                "Invalid TensorFlow tensor dimension");
        shape.push_back(dimension == UINT64_MAX ? -1 : int(dimension));
    }
    require(shape.size() <= 8 && (signature || !descriptor.integer(3)), "Unsupported TensorFlow tensor rank");
    return shape;
}

Tensor constant(const protobuf::Message& proto, Backend backend);

}  // namespace elman::io::tensorflow
