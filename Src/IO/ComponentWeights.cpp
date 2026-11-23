#include "IO/ComponentWeights.h"

#include <stdexcept>
#include <utility>

namespace elman::io {
namespace {

std::string weight_path(const std::filesystem::path& directory, const std::string& stem, DType dtype) {
  const std::filesystem::path regular = directory / (stem + ".safetensors");
  const std::filesystem::path half = directory / (stem + ".fp16.safetensors");
  if (dtype == DType::Float16 && std::filesystem::exists(half)) return half.string();
  if (std::filesystem::exists(regular)) return regular.string();
  if (std::filesystem::exists(half)) return half.string();
  return regular.string();
}

}  // namespace

ComponentWeights::ComponentWeights(const std::filesystem::path& directory, const std::string& stem, Backend backend, DType dtype)
    : file_(weight_path(directory, stem, dtype)), backend_(backend), dtype_(dtype) {
  if (!is_compute_float(dtype_)) throw std::invalid_argument("Checkpoint weights require a compute floating dtype");
}

Tensor ComponentWeights::get(const std::string& name, const Shape& shape) const {
  return get(name, shape, dtype_);
}

Tensor ComponentWeights::get(const std::string& name, const Shape& shape, DType dtype) const {
  const SafetensorInfo& info = file_.info(name);
  if (info.shape != shape || !is_compute_float(info.dtype)) throw std::invalid_argument("Incompatible checkpoint weight " + name);
  Tensor tensor = file_.load(name, backend_);
  return tensor.dtype() == dtype ? tensor : cast(tensor, dtype);
}

Tensor ComponentWeights::matrix(const std::string& name, int rows, int columns) const {
  if (file_.info(name).shape.size() == 4) return reshape(get(name, {rows, columns, 1, 1}), {rows, columns});
  return get(name, {rows, columns});
}

}  // namespace elman::io
