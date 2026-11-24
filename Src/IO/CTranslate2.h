#pragma once

#include "IO/Safetensors.h"

namespace elman {

// Reads the version 6 tensor container directly, without the CTranslate2 runtime.
// Tensor aliases share catalog entries; payloads are read on demand.
class CTranslate2File {
public:
    explicit CTranslate2File(const std::filesystem::path& path);
    const std::string& specification() const { return specification_; }
    uint32_t revision() const { return revision_; }
    const std::map<std::string, SafetensorInfo>& tensors() const { return tensors_; }
    const SafetensorInfo& info(const std::string& name) const;
    Tensor load(const std::string& name, Backend backend = Backend::CPU) const;
    // Integer weights use the reciprocal row scale stored at NAME_scale.
    Tensor load_float(const std::string& name, Backend backend, DType dtype) const;

private:
    std::filesystem::path path_;
    std::string specification_;
    uint32_t revision_ = 0;
    std::map<std::string, SafetensorInfo> tensors_;
};

} // namespace elman
