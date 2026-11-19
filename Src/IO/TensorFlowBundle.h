#pragma once

#include <map>

#include "IO/TensorFlowProto.h"

namespace elman::io::tensorflow {

class Bundle {
public:
    explicit Bundle(const std::filesystem::path& prefix);
    bool contains(const std::string& name) const { return entries_.count(name) != 0; }
    Tensor tensor(const std::string& name, Backend backend) const;

private:
    struct Entry {
        DType type;
        Shape shape;
        uint64_t shard, offset, size;
        uint32_t checksum;
    };
    std::filesystem::path prefix_;
    std::map<std::string, Entry> entries_;
    uint64_t shards_ = 0;
};

}  // namespace elman::io::tensorflow
