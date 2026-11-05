#include "IO/Checkpoint.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace elman {
namespace {

constexpr uint64_t hash_offset = 14695981039346656037ull;
constexpr uint64_t hash_prime = 1099511628211ull;
constexpr char magic[8] = {'E', 'L', 'M', 'A', 'N', '0', '1', '\0'};

class Writer {
public:
    explicit Writer(const std::string& path) : out_(path, std::ios::binary) {
        if (!out_) throw std::runtime_error("Cannot create checkpoint: " + path);
    }
    void byte(uint8_t value, bool hash = true) {
        out_.put(static_cast<char>(value));
        if (hash) { hash_ ^= value; hash_ *= hash_prime; }
    }
    void u32(uint32_t value) { for (int i = 0; i < 4; ++i) byte(static_cast<uint8_t>(value >> (8 * i))); }
    void u64(uint64_t value, bool hash = true) { for (int i = 0; i < 8; ++i) byte(static_cast<uint8_t>(value >> (8 * i)), hash); }
    void string(const std::string& text) {
        if (text.size() > 1048576) throw std::runtime_error("Checkpoint string is too long");
        u32(static_cast<uint32_t>(text.size()));
        for (uint8_t c : text) byte(c);
    }
    void values(const std::vector<float>& values) {
        u64(values.size());
        std::array<uint8_t, 65536> chunk;
        for (size_t begin = 0; begin < values.size(); begin += chunk.size() / 4) {
            size_t count = std::min(chunk.size() / 4, values.size() - begin);
            for (size_t i = 0; i < count; ++i) {
                float value = values[begin + i];
                if (!std::isfinite(value)) throw std::runtime_error("Cannot save non-finite checkpoint values");
                uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
                for (int b = 0; b < 4; ++b) chunk[4 * i + b] = uint8_t(bits >> (8 * b));
            }
            out_.write(reinterpret_cast<const char*>(chunk.data()), std::streamsize(count * 4));
            if (!out_) throw std::runtime_error("Checkpoint value write failed");
            for (size_t i = 0; i < count * 4; ++i) { hash_ ^= chunk[i]; hash_ *= hash_prime; }
        }
    }
    void finish() {
        u64(hash_, false);
        out_.flush();
        if (!out_) throw std::runtime_error("Checkpoint write failed");
        out_.close();
        if (!out_) throw std::runtime_error("Checkpoint close failed");
    }
private:
    std::ofstream out_;
    uint64_t hash_ = hash_offset;
};

class Reader {
public:
    explicit Reader(const std::string& path) : in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("Cannot open checkpoint: " + path);
    }
    uint8_t byte(bool hash = true) {
        int value = in_.get();
        if (value == std::char_traits<char>::eof()) throw std::runtime_error("Truncated checkpoint");
        if (hash) { hash_ ^= static_cast<uint8_t>(value); hash_ *= hash_prime; }
        return static_cast<uint8_t>(value);
    }
    uint32_t u32() { uint32_t value = 0; for (int i = 0; i < 4; ++i) value |= static_cast<uint32_t>(byte()) << (8 * i); return value; }
    uint64_t u64(bool hash = true) { uint64_t value = 0; for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(byte(hash)) << (8 * i); return value; }
    std::string string() {
        uint32_t size = u32();
        if (size > 1048576) throw std::runtime_error("Invalid checkpoint string length");
        std::string text(size, '\0');
        for (char& c : text) c = static_cast<char>(byte());
        return text;
    }
    std::vector<float> values(size_t expected) {
        uint64_t size = u64();
        if (size != expected) throw std::runtime_error("Checkpoint value count mismatch");
        std::vector<float> result(expected);
        std::array<uint8_t, 65536> chunk;
        for (size_t begin = 0; begin < expected; begin += chunk.size() / 4) {
            size_t count = std::min(chunk.size() / 4, expected - begin);
            in_.read(reinterpret_cast<char*>(chunk.data()), std::streamsize(count * 4));
            if (!in_) throw std::runtime_error("Truncated checkpoint values");
            for (size_t i = 0; i < count * 4; ++i) { hash_ ^= chunk[i]; hash_ *= hash_prime; }
            for (size_t i = 0; i < count; ++i) {
                uint32_t bits = 0;
                for (int b = 0; b < 4; ++b) bits |= uint32_t(chunk[4 * i + b]) << (8 * b);
                std::memcpy(&result[begin + i], &bits, sizeof(float));
                if (!std::isfinite(result[begin + i])) throw std::runtime_error("Non-finite checkpoint value");
            }
        }
        return result;
    }
    void finish() {
        uint64_t expected = hash_;
        if (u64(false) != expected) throw std::runtime_error("Checkpoint checksum mismatch");
        if (in_.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected trailing checkpoint bytes");
    }
private:
    std::ifstream in_;
    uint64_t hash_ = hash_offset;
};

void replace_file(const std::filesystem::path& source, const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot replace checkpoint (Windows error " + std::to_string(GetLastError()) + ")");
#else
    std::filesystem::rename(source, destination);
#endif
}

}

void save_checkpoint(const std::string& path, const Model& model, const Json& definition, const TrainingState& state) {
    if (state.first_moment.size() != model.parameters().size() || state.second_moment.size() != model.parameters().size())
        throw std::runtime_error("Checkpoint optimizer state count mismatch");
    auto destination = std::filesystem::path(path);
    if (!destination.parent_path().empty()) std::filesystem::create_directories(destination.parent_path());
    auto temporary = destination; temporary += ".tmp";
    try {
        Writer writer(temporary.string());
        for (uint8_t c : magic) writer.byte(c);
        writer.u32(1);
        writer.string(model.name());
        writer.string(definition.dump(0));
        writer.u64(state.step);
        writer.string(state.optimizer);
        writer.u32(static_cast<uint32_t>(model.parameters().size()));
        for (size_t i = 0; i < model.parameters().size(); ++i) {
            const auto& parameter = model.parameters()[i];
            if (state.first_moment[i].size() != parameter.value.numel() || state.second_moment[i].size() != parameter.value.numel())
                throw std::runtime_error("Checkpoint optimizer shape mismatch: " + parameter.name);
            writer.string(parameter.name);
            writer.u32(static_cast<uint32_t>(parameter.value.shape().size()));
            for (int dim : parameter.value.shape()) writer.u32(static_cast<uint32_t>(dim));
            writer.values(parameter.value.data());
            writer.values(state.first_moment[i]);
            writer.values(state.second_moment[i]);
        }
        writer.finish();
        replace_file(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void load_checkpoint(const std::string& path, Model& model, const Json& definition, TrainingState* state) {
    Reader reader(path);
    for (uint8_t c : magic) if (reader.byte() != c) throw std::runtime_error("Invalid checkpoint signature");
    if (reader.u32() != 1) throw std::runtime_error("Unsupported checkpoint version");
    if (reader.string() != model.name()) throw std::runtime_error("Checkpoint model name mismatch");
    if (reader.string() != definition.dump(0)) throw std::runtime_error("Checkpoint model definition mismatch");
    TrainingState loaded;
    loaded.step = reader.u64();
    if (loaded.step > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) throw std::runtime_error("Checkpoint training step out of range");
    loaded.optimizer = reader.string();
    if (loaded.optimizer != "adam" && loaded.optimizer != "adamw" && loaded.optimizer != "sgd") throw std::runtime_error("Invalid checkpoint optimizer");
    if (reader.u32() != model.parameters().size()) throw std::runtime_error("Checkpoint parameter count mismatch");
    std::vector<std::vector<float>> parameters;
    for (const auto& parameter : model.parameters()) {
        if (reader.string() != parameter.name) throw std::runtime_error("Checkpoint parameter name mismatch: " + parameter.name);
        if (reader.u32() != parameter.value.shape().size()) throw std::runtime_error("Checkpoint parameter rank mismatch: " + parameter.name);
        for (int dim : parameter.value.shape()) if (reader.u32() != static_cast<uint32_t>(dim)) throw std::runtime_error("Checkpoint parameter shape mismatch: " + parameter.name);
        parameters.push_back(reader.values(parameter.value.numel()));
        loaded.first_moment.push_back(reader.values(parameter.value.numel()));
        loaded.second_moment.push_back(reader.values(parameter.value.numel()));
    }
    reader.finish();
    for (size_t i = 0; i < parameters.size(); ++i) model.parameters()[i].value.assign(parameters[i]);
    if (state) *state = std::move(loaded);
}

}
