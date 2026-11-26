#include <algorithm>
#include <stdexcept>

#include "Models/Model.h"

namespace elman {
namespace {
Json include_file(const std::filesystem::path& path,
                  std::vector<std::filesystem::path>& stack);

Json expand_includes(const Json& value, const std::filesystem::path& directory,
                     std::vector<std::filesystem::path>& stack, int depth) {
  if (depth > 128) {
    throw std::invalid_argument("Model JSON nesting limit exceeded");
  }
  if (value.is_array()) {
    Json::Array result;
    for (const auto& item : value.array()) {
      result.push_back(expand_includes(item, directory, stack, depth + 1));
    }
    return result;
  }
  if (!value.is_object()) {
    return value;
  }
  if (value.contains("$include")) {
    if (value.object().size() != 1 || value.at("$include").string().empty()) {
      throw std::invalid_argument(
          "$include requires a nonempty path and no sibling fields");
    }
    return include_file(
        directory / std::filesystem::u8path(value.at("$include").string()),
        stack);
  }
  Json::Object result;
  for (const auto& item : value.object()) {
    result[item.first] =
        expand_includes(item.second, directory, stack, depth + 1);
  }
  return result;
}

Json include_file(const std::filesystem::path& path,
                  std::vector<std::filesystem::path>& stack) {
  const auto canonical = std::filesystem::canonical(path);
  if (stack.size() >= 32 ||
      std::find(stack.begin(), stack.end(), canonical) != stack.end()) {
    throw std::invalid_argument(
        "Cyclic or excessively deep model JSON include: " + path.u8string());
  }
  stack.push_back(canonical);
  Json result =
      expand_includes(Json::read(canonical), canonical.parent_path(), stack, 0);
  stack.pop_back();
  return result;
}
}  // namespace

Json read_model_definition(const std::filesystem::path& path) {
  std::vector<std::filesystem::path> stack;
  return include_file(path, stack);
}
}  // namespace elman
