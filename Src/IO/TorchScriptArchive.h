#pragma once

#include <map>
#include <memory>

#include "IO/ZipArchive.h"
#include "Core/Tensor.h"

namespace elman::io {

struct TorchScriptValue {
    enum class Kind { None, Boolean, Integer, Number, String, Tensor, List, Object };
    Kind kind = Kind::None;
    int64_t integer = 0;
    double number = 0;
    std::string text;
    elman::Tensor tensor;
    std::vector<std::shared_ptr<const TorchScriptValue>> sequence;
    std::map<std::string, std::shared_ptr<const TorchScriptValue>> fields;
};

struct TorchScriptArchive {
    TorchScriptArchive(const std::filesystem::path& path, Backend backend);
    std::shared_ptr<const TorchScriptValue> module, constants;
    std::map<std::string, std::string> source;
};

}  // namespace elman::io
