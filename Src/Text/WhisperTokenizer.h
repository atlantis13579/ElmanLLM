#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

namespace elman {

class WhisperTokenizer {
public:
    explicit WhisperTokenizer(const std::filesystem::path& directory);
    int32_t token(const std::string& spelling) const;
    int32_t end() const { return end_; }
    int32_t timestamp_begin() const { return timestamp_; }
    size_t size() const { return bytes_.size(); }
    const std::string& bytes(int32_t id) const;
    std::string decode(const std::vector<int32_t>& ids) const;
private:
    std::vector<std::string> bytes_;
    std::map<std::string, int32_t> tokens_;
    int32_t end_ = -1, timestamp_ = -1;
};

} // namespace elman
