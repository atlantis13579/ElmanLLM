#include "Arguments.h"
#include "ToolIO.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "Backends/CpuParallel.h"
#include "Core/Float16.h"
#include "Core/TensorDispatch.h"
#include "IO/AtomicFile.h"
#include "IO/Safetensors.h"

namespace {
namespace fs = std::filesystem;
using namespace elman;

struct Options {
    fs::path source, destination;
    int group_size = 64, chunk_rows = 128, threads = 4;
};

void usage() {
    std::cout << "Usage: elman_quantize_weights --model DIR --output NEW_DIR\n"
                 "       [--group-size N] [--chunk-rows N] [--threads N]\n";
}

Options parse(const std::vector<std::string>& arguments) {
    Options options;
    for (size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view argument = arguments[i];
        const auto equals = argument.find('=');
        const auto key = argument.substr(0, equals);
        if (equals == std::string_view::npos && ++i == arguments.size())
            throw std::invalid_argument("Missing value for " + std::string(key));
        const auto value = equals == std::string_view::npos ? std::string_view(arguments[i]) : argument.substr(equals + 1);
        if (key == "--model") options.source = fs::u8path(value);
        else if (key == "--output") options.destination = fs::u8path(value);
        else if (key == "--group-size") options.group_size = tools::integer_argument(value, key);
        else if (key == "--chunk-rows") options.chunk_rows = tools::integer_argument(value, key);
        else if (key == "--threads") options.threads = tools::integer_argument(value, key);
        else throw std::invalid_argument("Unknown option " + std::string(key));
    }
    if (options.source.empty() || options.destination.empty() || options.group_size <= 0 ||
        options.chunk_rows <= 0 || options.chunk_rows % 2 || options.threads <= 0 || options.threads > 256)
        throw std::invalid_argument("Provide model and output directories, a positive group size, positive even chunk rows, and 1-256 threads");
    options.source = fs::canonical(options.source);
    options.destination = fs::weakly_canonical(fs::absolute(options.destination));
    return options;
}

Json::Array json_shape(const Shape& shape) {
    return Json::Array(shape.begin(), shape.end());
}

void write_bytes(std::ostream& out, const uint8_t* bytes, size_t count) {
    if (count > size_t(std::numeric_limits<std::streamsize>::max())) throw std::overflow_error("Output block is too large");
    if (count) out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(count));
    if (!out) throw std::runtime_error("Cannot write weight payload");
}

void copy_weight(std::ofstream& out, const SafetensorsWeights& source, const std::string& name, uint64_t bytes) {
    std::vector<uint8_t> buffer(static_cast<size_t>(std::min<uint64_t>(bytes, 1024 * 1024)));
    for (uint64_t offset = 0; offset < bytes;) {
        const auto count = static_cast<size_t>(std::min<uint64_t>(buffer.size(), bytes - offset));
        source.read_bytes(name, offset, buffer.data(), count);
        write_bytes(out, buffer.data(), count);
        offset += count;
    }
}

struct Errors {
    double signal = 0, squared = 0;
    float maximum = 0;

    void add(const Errors& other) {
        signal += other.signal;
        squared += other.squared;
        maximum = std::max(maximum, other.maximum);
    }
};

uint8_t quantize(float value, float inverse_scale) {
    const float scaled = value * inverse_scale;
    if (!std::isfinite(scaled)) return value == 0.f ? 8 : value < 0.f ? 0 : 15;
    if (scaled <= -8.f) return 0;
    if (scaled >= 7.f) return 15;
    const int lower = static_cast<int>(std::floor(scaled));
    const float fraction = scaled - lower;
    return static_cast<uint8_t>(8 + lower + (fraction > .5f || (fraction == .5f && lower % 2 != 0)));
}

Errors quantize_matrix(std::ofstream& out, const SafetensorsWeights& source,
                       const std::string& name, const Options& options, int rows, int columns, int groups, uint64_t code_bytes) {
    const auto payload = out.tellp();
    Errors errors;
    for (int start = 0; start < rows;) {
        const int count = std::min(options.chunk_rows, rows - start);
        const auto values = source.load_slice(name, 0, start, count, Backend::CPU).data();
        std::vector<uint8_t> codes(values.size()), scales(size_t(count) * groups * 2);
        std::vector<Errors> by_row(static_cast<size_t>(count));
        parallel_for_cpu(count, 1, [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                const auto base = static_cast<size_t>(row) * columns;
                auto& metrics = by_row[static_cast<size_t>(row)];
                for (int group = 0; group < groups; ++group) {
                    const int first = group * options.group_size;
                    const int last = first + std::min(options.group_size, columns - first);
                    float maximum = 0;
                    for (int col = first; col < last; ++col) {
                        const float value = values[base + col];
                        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite source weight: " + name);
                        maximum = std::max(maximum, std::abs(value));
                    }
                    const float proposed = maximum / 7.f;
                    const uint16_t encoded = numeric::float_to_bfloat(proposed == 0.f ? 1.f : proposed);
                    const float scale = numeric::bfloat_to_float(encoded);
                    if (!std::isfinite(scale) || !(scale > 0.f))
                        throw std::runtime_error("BF16 scale cannot represent source range: " + name);
                    const auto scale_position = (static_cast<size_t>(row) * groups + group) * 2;
                    scales[scale_position] = static_cast<uint8_t>(encoded);
                    scales[scale_position + 1] = static_cast<uint8_t>(encoded >> 8);
                    const float inverse_scale = 1.f / scale;
                    for (int col = first; col < last; ++col) {
                        const float value = values[base + col];
                        const auto code = quantize(value, inverse_scale);
                        codes[base + col] = code;
                        const float reconstructed = numeric::bfloat_to_float(numeric::float_to_bfloat((int(code) - 8) * scale));
                        const float difference = reconstructed - value;
                        if (!std::isfinite(difference)) throw std::runtime_error("Nonfinite reconstruction error: " + name);
                        metrics.signal += double(value) * value;
                        metrics.squared += double(difference) * difference;
                        metrics.maximum = std::max(metrics.maximum, std::abs(difference));
                    }
                }
            }
        });
        const auto packed_size = (codes.size() + 1) / 2;
        for (size_t i = 0; i < packed_size; ++i) {
            const auto high = 2 * i + 1 < codes.size() ? codes[2 * i + 1] : 0;
            codes[i] = static_cast<uint8_t>(codes[2 * i] | (high << 4));
        }
        out.seekp(payload + static_cast<std::streamoff>(uint64_t(start) * columns / 2));
        write_bytes(out, codes.data(), packed_size);
        out.seekp(payload + static_cast<std::streamoff>(code_bytes + uint64_t(start) * groups * 2));
        write_bytes(out, scales.data(), scales.size());
        for (const auto& metrics : by_row) errors.add(metrics);
        start += count;
    }
    const uint8_t zero_point[] = {8, 0, 0, 0};
    write_bytes(out, zero_point, sizeof(zero_point));
    return errors;
}

void export_weights(const Options& options) {
    if (fs::exists(options.destination)) throw std::invalid_argument("Choose a new output directory");
    const auto config = Json::read(options.source / "config.json");
    const auto model = config.at("model_type").string();
    if (model != "qwen2" && model != "deepseek_v2" && model != "deepseek_v3")
        throw std::invalid_argument("Unsupported decoder weight layout: " + model);
    if (!fs::exists(options.source / "model.safetensors.index.json")) {
        const auto path = fs::canonical(options.source / "model.safetensors");
        if (path.parent_path() != options.source || !fs::is_regular_file(path))
            throw std::invalid_argument("Source weights must be a file directly inside the model directory");
    }
    const SafetensorsWeights source(options.source);
    if (source.tensors().empty()) throw std::invalid_argument("No source weights found");
    const auto binary_hash = tools::sha256_file(tools::executable_path());
    fs::create_directories(options.destination.parent_path());
    if (!fs::create_directory(options.destination)) throw std::runtime_error("Output directory already exists");

    Json::Object weight_map, quantized, files, metrics;
    uint64_t total_size = 0;
    const auto manifest_path = options.destination / "export-manifest.json";
    Json::Object manifest{{"status", "in-progress"}, {"source", options.source.u8string()},
                         {"generator", "elman_quantize_weights"}, {"generator_sha256", binary_hash},
                         {"group_size", options.group_size}};
    auto save_manifest = [&] {
        manifest["files"] = files;
        manifest["weights"] = metrics;
        tools::write_json_atomically(manifest_path, manifest);
    };
    save_manifest();

    size_t ordinal = 0;
    for (const auto& [name, info] : source.tensors()) {
        if (!is_compute_float(info.dtype)) throw std::invalid_argument("Unsupported decoder weight dtype: " + name);
        std::ostringstream numbered;
        numbered << "weights-" << std::setfill('0') << std::setw(5) << ordinal << ".safetensors";
        const auto filename = numbered.str();
        const auto path = options.destination / filename;
        const std::string gate = ".mlp.gate.weight";
        const bool matrix = info.shape.size() == 2 &&
            (name.size() < gate.size() || name.compare(name.size() - gate.size(), gate.size(), gate) != 0);
        const auto elements = shape_element_count(info.shape);
        Json::Object record{{"quantized", matrix}, {"elements", double(elements)}};
        std::vector<SafetensorSpec> entries;
        if (!matrix) {
            entries.push_back({name, info.dtype, info.shape});
            write_file_atomically(path, [&](const auto& pending) {
                std::ofstream out(pending, std::ios::binary);
                write_safetensors_header(out, entries, {{"format", "pt"}});
                copy_weight(out, source, name, info.bytes);
                out.close();
                if (!out) throw std::runtime_error("Cannot close shard: " + filename);
            });
        } else {
            const int rows = info.shape[0], columns = info.shape[1];
            if (rows <= 0 || columns <= 0) throw std::invalid_argument("Decoder matrix dimensions must be positive: " + name);
            const int groups = 1 + (columns - 1) / options.group_size;
            const uint64_t code_bytes = (uint64_t(elements) + 1) / 2;
            if (code_bytes > uint64_t(std::numeric_limits<int>::max())) throw std::overflow_error("Packed matrix is too large: " + name);
            const auto code = name + ".codes", scale = name + ".scales", zero = name + ".zero_points";
            entries = {{code, DType::UInt8, {static_cast<int>(code_bytes)}},
                       {scale, DType::BFloat16, {rows, groups}}, {zero, DType::Int32, {}}};
            Errors errors;
            write_file_atomically(path, [&](const auto& pending) {
                std::ofstream out(pending, std::ios::binary);
                write_safetensors_header(out, entries, {{"format", "pt"}});
                errors = quantize_matrix(out, source, name, options, rows, columns, groups, code_bytes);
                out.close();
                if (!out) throw std::runtime_error("Cannot close shard: " + filename);
            });
            quantized[name] = Json::Object{{"shape", json_shape(info.shape)}, {"dtype", "quint4"},
                {"group_size", options.group_size}, {"codes", code}, {"scales", scale}, {"zero_points", zero}};
            record["squared_signal"] = errors.signal;
            record["squared_error"] = errors.squared;
            record["max_error"] = double(errors.maximum);
        }
        for (const auto& entry : entries) {
            if (!weight_map.emplace(entry.name, filename).second) throw std::runtime_error("Duplicate exported weight: " + entry.name);
            const auto bytes = dtype_storage_bytes(entry.dtype, shape_element_count(entry.shape));
            if (bytes > 9007199254740991ull - total_size) throw std::overflow_error("Export size exceeds exact JSON integer range");
            total_size += bytes;
        }
        files[filename] = Json::Object{{"bytes", double(fs::file_size(path))}, {"sha256", tools::sha256_file(path)}};
        metrics[name] = std::move(record);
        save_manifest();
        std::cout << ++ordinal << '/' << source.tensors().size() << ' ' << name << std::endl;
    }
    for (auto name : {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "added_tokens.json", "LICENSE"}) {
        if (fs::is_regular_file(options.source / name)) fs::copy_file(options.source / name, options.destination / name);
    }
    tools::write_json_atomically(options.destination / "model.safetensors.index.json",
        Json::Object{{"metadata", Json::Object{{"total_size", double(total_size)}}}, {"weight_map", std::move(weight_map)}});
    const auto quantized_count = quantized.size();
    tools::write_json_atomically(options.destination / "quantization.json", Json::Object{{"format", "elman-affine-v1"},
        {"algorithm", "groupwise-symmetric-rtn"}, {"group_size", options.group_size}, {"tensors", std::move(quantized)}});
    manifest["status"] = "exported-unvalidated";
    manifest["logical_tensors"] = double(source.tensors().size());
    manifest["quantized_tensors"] = double(quantized_count);
    manifest["payload_bytes"] = double(total_size);
    save_manifest();
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        std::vector<std::string> arguments;
        for (int i = 1; i < argc; ++i) arguments.push_back(tools::utf8_argument(argv[i]));
        if (arguments.size() == 1 && (arguments.front() == "--help" || arguments.front() == "-h")) {
            usage();
            return 0;
        }
        const auto options = parse(arguments);
        set_cpu_thread_count(options.threads);
        export_weights(options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Elman quantization: " << error.what() << '\n';
        return 1;
    }
}
