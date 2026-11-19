#pragma once

#include "IO/Safetensors.h"

namespace elman::io {

class ComponentWeights {
 public:
  ComponentWeights(const std::filesystem::path& directory, const std::string& stem, Backend backend, DType dtype);
  Tensor get(const std::string& name, const Shape& shape) const;
  Tensor get(const std::string& name, const Shape& shape, DType dtype) const;
  Tensor matrix(const std::string& name, int rows, int columns) const;
  bool contains(const std::string& name) const { return file_.tensors().count(name) != 0; }

 private:
  SafetensorsFile file_;
  Backend backend_;
  DType dtype_;
};

}  // namespace elman::io
