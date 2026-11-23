#include "IO/Checkpoint.h"
#include "Core/Device.h"
#include "IO/AtomicFile.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {

constexpr uint64_t hash_offset = 14695981039346656037ull;
constexpr uint64_t hash_prime = 1099511628211ull;
constexpr char magic[8] = {'E', 'L', 'M', 'A', 'N', '0', '1', '\0'};

bool matches_model_definition(const Model& model, const Json& definition, const std::string& stored) {
    const std::string current = definition.dump(0);
    if (stored == current) return true;
    if (model.architecture() != "spatial_policy_value") return false;

    // This fully convolutional model has no parameters or optimizer slots indexed by height or width.
    Json::Object saved = Json::parse(stored).object();
    const Json::Array& shape = saved.at("input_shape").array();
    const Json::Array& expected = definition.at("input_shape").array();
    if (shape.size() != 3 || expected.size() != 3 || shape[0].integer() != expected[0].integer()) return false;
    const int height = shape[1].integer(), width = shape[2].integer();
    if (height < 1 || width < 1 ||
        int64_t(height) * width + definition.at("global_actions").integer() + 1 > std::numeric_limits<int>::max())
        return false;
    saved.at("input_shape") = definition.at("input_shape");
    return Json(std::move(saved)).dump(0) == current;
}

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
    void values(const std::vector<double>& values) {
        u64(values.size());
        std::array<uint8_t, 65536> chunk;
        for (size_t begin = 0; begin < values.size(); begin += chunk.size() / 8) {
            size_t count = std::min(chunk.size() / 8, values.size() - begin);
            for (size_t i = 0; i < count; ++i) {
                double value = values[begin + i];
                if (!std::isfinite(value)) throw std::runtime_error("Cannot save non-finite checkpoint values");
                uint64_t bits; std::memcpy(&bits, &value, sizeof(bits));
                for (int b = 0; b < 8; ++b) chunk[8 * i + b] = uint8_t(bits >> (8 * b));
            }
            out_.write(reinterpret_cast<const char*>(chunk.data()), std::streamsize(count * 8));
            if (!out_) throw std::runtime_error("Checkpoint value write failed");
            for (size_t i = 0; i < count * 8; ++i) { hash_ ^= chunk[i]; hash_ *= hash_prime; }
        }
    }
    void raw_values(const std::vector<uint8_t>& values) {
        u64(values.size());
        if (!values.empty()) out_.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()));
        if (!out_) throw std::runtime_error("Checkpoint raw value write failed");
        for (uint8_t value:values) {hash_^=value;hash_*=hash_prime;}
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
    std::vector<double> double_values(size_t expected) {
        uint64_t size = u64();
        if (size != expected) throw std::runtime_error("Checkpoint value count mismatch");
        std::vector<double> result(expected);
        std::array<uint8_t, 65536> chunk;
        for (size_t begin = 0; begin < expected; begin += chunk.size() / 8) {
            size_t count = std::min(chunk.size() / 8, expected - begin);
            in_.read(reinterpret_cast<char*>(chunk.data()), std::streamsize(count * 8));
            if (!in_) throw std::runtime_error("Truncated checkpoint values");
            for (size_t i = 0; i < count * 8; ++i) { hash_ ^= chunk[i]; hash_ *= hash_prime; }
            for (size_t i = 0; i < count; ++i) {
                uint64_t bits = 0;
                for (int b = 0; b < 8; ++b) bits |= uint64_t(chunk[8 * i + b]) << (8 * b);
                std::memcpy(&result[begin + i], &bits, sizeof(double));
                if (!std::isfinite(result[begin + i])) throw std::runtime_error("Non-finite checkpoint value");
            }
        }
        return result;
    }
    std::vector<uint8_t> raw_values(size_t expected) {
        if (u64()!=expected) throw std::runtime_error("Checkpoint raw value count mismatch");
        std::vector<uint8_t> result(expected);
        if (expected) in_.read(reinterpret_cast<char*>(result.data()),std::streamsize(expected));
        if (!in_) throw std::runtime_error("Truncated checkpoint raw values");
        for (uint8_t value:result) {hash_^=value;hash_*=hash_prime;}
        return result;
    }
    uint64_t finish() {
        uint64_t expected = hash_;
        if (u64(false) != expected) throw std::runtime_error("Checkpoint checksum mismatch");
        if (in_.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected trailing checkpoint bytes");
        return expected;
    }
private:
    std::ifstream in_;
    uint64_t hash_ = hash_offset;
};

void finite_parameter(const Tensor& value) {
    if (is_complex(value.dtype())) {
        for (const std::complex<double> z : value.complex_data()) if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
            throw std::runtime_error("Non-finite complex checkpoint parameter");
    } else if (value.dtype()==DType::Float64) {
        for (double x:value.double_data()) if (!std::isfinite(x)) throw std::runtime_error("Non-finite checkpoint parameter");
    } else for (float x:value.data()) if (!std::isfinite(x)) throw std::runtime_error("Non-finite checkpoint parameter");
}

void tensor_moment(const Tensor& moment,const Tensor& value,DType dtype,bool second) {
    if (!moment.defined() || moment.dtype()!=dtype || moment.shape()!=value.shape() ||
        moment.backend()!=value.backend() || moment.requires_grad()) throw std::runtime_error("Checkpoint optimizer moment descriptor mismatch");
    if (is_complex(dtype)) {
        for (const std::complex<double> z : moment.complex_data())
            if (!std::isfinite(z.real()) || !std::isfinite(z.imag()) || (second && (z.real() < 0. || z.imag() < 0.)))
                throw std::runtime_error("Invalid complex checkpoint optimizer moment");
    } else if (dtype==DType::Float64) {
        for(double x:moment.double_data()) if(!std::isfinite(x)||(second&&x<0.))throw std::runtime_error("Invalid checkpoint optimizer moment");
    } else for(float x:moment.data()) if(!std::isfinite(x)||(second&&x<0.f))throw std::runtime_error("Invalid checkpoint optimizer moment");
}

}

void save_checkpoint(const std::string& path, const Model& model, const Json& definition,
                     const TrainingState& state, const TrainingProgress* progress) {
    if (state.first_moment.size() != model.parameters().size() || state.second_moment.size() != model.parameters().size())
        throw std::runtime_error("Checkpoint optimizer state count mismatch");
    bool wide = false, low = false, complex_parameters = false;
    const size_t count = model.parameters().size();
    if ((!state.first_moment_double.empty() && state.first_moment_double.size() != count) ||
        (!state.second_moment_double.empty() && state.second_moment_double.size() != count) ||
        (!state.first_moment_float.empty() && state.first_moment_float.size()!=count) ||
        (!state.second_moment_float.empty() && state.second_moment_float.size()!=count) ||
        (!state.sgd_buffer_initialized.empty() && state.sgd_buffer_initialized.size()!=count))
        throw std::runtime_error("Checkpoint tensor optimizer state count mismatch");
    if (state.step > uint64_t(std::numeric_limits<int64_t>::max()) ||
        (state.optimizer != "sgd" && state.optimizer != "adam" && state.optimizer != "adamw"))
        throw std::runtime_error("Invalid checkpoint optimizer metadata");
    const std::vector<Parameter> buffers = model.checkpoint_buffers();
    if (buffers.size() > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("Too many checkpoint buffers");
    for (size_t i = 0; i < count; ++i) {
        const Tensor& value = model.parameters()[i].value;
        const bool parameter_low=is_low_precision_float(value.dtype());
        const bool resident = state.uses_resident_float32(value);
        if (!is_arithmetic_float(value.dtype()) && !is_complex(value.dtype()) && !parameter_low) throw std::runtime_error("Checkpoint requires floating parameters");
        finite_parameter(value);
        if (!parameter_low && !resident && !is_complex(value.dtype()) && ((!state.first_moment_float.empty()&&state.first_moment_float[i].defined()) ||
            (!state.second_moment_float.empty()&&state.second_moment_float[i].defined()) ||
            (!state.sgd_buffer_initialized.empty()&&state.sgd_buffer_initialized[i]!=0))) throw std::runtime_error("Unexpected low-format checkpoint state");
        if (is_complex(value.dtype())) {
            complex_parameters = true;
            if (!state.first_moment[i].empty() || !state.second_moment[i].empty() ||
                state.first_moment_double.size() != count || state.second_moment_double.size() != count ||
                (!state.first_moment_float.empty() && state.first_moment_float[i].defined()) ||
                (!state.second_moment_float.empty() && state.second_moment_float[i].defined()) ||
                state.sgd_buffer_initialized.size() != count || state.sgd_buffer_initialized[i] > 1 ||
                (state.optimizer != "sgd" && state.sgd_buffer_initialized[i]))
                throw std::runtime_error("Invalid complex checkpoint optimizer storage");
            tensor_moment(state.first_moment_double[i], value, value.dtype(), false);
            tensor_moment(state.second_moment_double[i], value, value.dtype(), true);
        } else if (value.dtype() == DType::Float64) {
            wide = true;
            if (!state.first_moment[i].empty() || !state.second_moment[i].empty() ||
                state.first_moment_double.size() != count || state.second_moment_double.size() != count)
                throw std::runtime_error("Checkpoint Double optimizer storage mismatch");
            tensor_moment(state.first_moment_double[i],value,DType::Float64,false);
            tensor_moment(state.second_moment_double[i],value,DType::Float64,true);
        } else if (parameter_low) {
            low=true;
            if (!state.first_moment[i].empty() || !state.second_moment[i].empty() ||
                (!state.first_moment_double.empty()&&state.first_moment_double[i].defined()) ||
                (!state.second_moment_double.empty()&&state.second_moment_double[i].defined()) ||
                state.first_moment_float.size()!=count || state.second_moment_float.size()!=count ||
                state.sgd_buffer_initialized.size()!=count || state.sgd_buffer_initialized[i]>1 ||
                (state.optimizer!="sgd"&&state.sgd_buffer_initialized[i]!=0)) throw std::runtime_error("Invalid low-format checkpoint state");
            tensor_moment(state.first_moment_float[i],value,DType::Float32,false);
            tensor_moment(state.second_moment_float[i],value,DType::Float32,true);
        } else if (resident) {
            if (!state.first_moment[i].empty() || !state.second_moment[i].empty() ||
                state.first_moment_float.size() != count || state.second_moment_float.size() != count ||
                (!state.first_moment_double.empty() && state.first_moment_double[i].defined()) ||
                (!state.second_moment_double.empty() && state.second_moment_double[i].defined()) ||
                (!state.sgd_buffer_initialized.empty() && state.sgd_buffer_initialized[i] != 0)) {
                throw std::runtime_error("Checkpoint resident optimizer state mismatch");
            }
            tensor_moment(state.first_moment_float[i], value, DType::Float32, false);
            tensor_moment(state.second_moment_float[i], value, DType::Float32, true);
        } else {
            if (state.first_moment[i].size() != value.numel() || state.second_moment[i].size() != value.numel() ||
                (!state.first_moment_double.empty() && state.first_moment_double[i].defined()) ||
                (!state.second_moment_double.empty() && state.second_moment_double[i].defined()))
                throw std::runtime_error("Checkpoint optimizer shape mismatch");
        }
    }
    write_file_atomically(path, [&](const std::filesystem::path& temporary) {
        Writer writer(temporary.string());
        for (uint8_t c : magic) writer.byte(c);
        // Version 4 adds complex tensor moments and component-wise SGD history.
        // Real-only models retain their established version and byte layout.
        const bool raw_parameters = low || complex_parameters;
        const uint32_t parameter_version = complex_parameters ? 4 : low ? 3 : wide ? 2 : 1;
        writer.u32(buffers.empty() ? (progress ? 6 : parameter_version) : 7);
        if (progress || !buffers.empty()) writer.u32(parameter_version);
        if (!buffers.empty()) writer.u32(progress ? 1 : 0);
        writer.string(model.name());
        writer.string(definition.dump(0));
        writer.u64(state.step);
        writer.string(state.optimizer);
        writer.u32(static_cast<uint32_t>(model.parameters().size()));
        for (size_t i = 0; i < model.parameters().size(); ++i) {
            const Parameter& parameter = model.parameters()[i];
            writer.string(parameter.name);
            writer.u32(static_cast<uint32_t>(parameter.value.shape().size()));
            for (int dim : parameter.value.shape()) writer.u32(static_cast<uint32_t>(dim));
            if (wide || raw_parameters) writer.u32(uint32_t(parameter.value.dtype()));
            if (raw_parameters) writer.raw_values(parameter.value.raw_data());
            if (is_complex(parameter.value.dtype())) {
                writer.u32(state.sgd_buffer_initialized[i]);
                writer.raw_values(state.first_moment_double[i].raw_data());
                writer.raw_values(state.second_moment_double[i].raw_data());
            } else if (parameter.value.dtype() == DType::Float64) {
                if (!raw_parameters) writer.values(parameter.value.double_data());
                writer.values(state.first_moment_double[i].double_data());
                writer.values(state.second_moment_double[i].double_data());
            } else if (is_low_precision_float(parameter.value.dtype())) {
                writer.u32(state.sgd_buffer_initialized[i]);
                writer.values(state.first_moment_float[i].data());
                writer.values(state.second_moment_float[i].data());
            } else {
                if (!raw_parameters) writer.values(parameter.value.data());
                if (state.uses_resident_float32(parameter.value)) {
                    writer.values(state.first_moment_float[i].data());
                    writer.values(state.second_moment_float[i].data());
                } else {
                    writer.values(state.first_moment[i]);
                    writer.values(state.second_moment[i]);
                }
            }
        }
        if (!buffers.empty()) {
            writer.u32(static_cast<uint32_t>(buffers.size()));
            for (const Parameter& buffer : buffers) {
                writer.string(buffer.name);
                writer.u32(static_cast<uint32_t>(buffer.value.dtype()));
                writer.u32(static_cast<uint32_t>(buffer.value.shape().size()));
                for (int dim : buffer.value.shape()) writer.u32(static_cast<uint32_t>(dim));
                writer.raw_values(buffer.value.raw_data());
            }
        }
        if (progress) {
            const SamplerState& sampler = progress->sampler;
            if (progress->configuration.empty() || sampler.offset >= sampler.order.size() || !sampler.random_state)
                throw std::invalid_argument("Invalid training progress");
            writer.string(progress->configuration);
            writer.u64(sampler.order.size());
            writer.u64(sampler.offset);
            writer.u32(sampler.random_state);
            for (size_t index : sampler.order) writer.u64(index);
            if (progress->attempted_steps < state.step || progress->attempted_steps > uint64_t(std::numeric_limits<int64_t>::max()) ||
                !std::isfinite(progress->loss_scale.scale) || progress->loss_scale.scale <= 0)
                throw std::invalid_argument("Invalid optimizer or loss scale progress");
            writer.u64(progress->attempted_steps);
            uint32_t scale_bits;
            std::memcpy(&scale_bits, &progress->loss_scale.scale, sizeof(scale_bits));
            writer.u32(scale_bits);
            writer.u32(progress->loss_scale.finite_steps);
        }
        writer.finish();
    });
}

void load_checkpoint(const std::string& path, Model& model, const Json& definition,
                     TrainingState* state, TrainingProgress* progress, size_t expected_samples) {
    Reader reader(path);
    for (uint8_t c : magic) if (reader.byte() != c) throw std::runtime_error("Invalid checkpoint signature");
    const uint32_t envelope = reader.u32();
    const bool has_buffers = envelope == 7;
    const uint32_t version = envelope >= 5 && envelope <= 7 ? reader.u32() : envelope;
    if (version < 1 || version > 4) throw std::runtime_error("Unsupported checkpoint version");
    const uint32_t progress_flag = has_buffers ? reader.u32() : uint32_t(envelope == 5 || envelope == 6);
    if (progress_flag > 1) throw std::runtime_error("Invalid checkpoint training progress flag");
    const bool has_progress = progress_flag != 0;
    if (reader.string() != model.name()) throw std::runtime_error("Checkpoint model name mismatch");
    if (!matches_model_definition(model, definition, reader.string())) throw std::runtime_error("Checkpoint model definition mismatch");
    TrainingState loaded;
    loaded.resident_float32 = state && state->resident_float32;
    loaded.step = reader.u64();
    if (loaded.step > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) throw std::runtime_error("Checkpoint training step out of range");
    loaded.optimizer = reader.string();
    if (loaded.optimizer != "adam" && loaded.optimizer != "adamw" && loaded.optimizer != "sgd") throw std::runtime_error("Invalid checkpoint optimizer");
    if (reader.u32() != model.parameters().size()) throw std::runtime_error("Checkpoint parameter count mismatch");
    std::vector<std::vector<uint8_t>> parameters;
    for (const Parameter& parameter : model.parameters()) {
        if (reader.string() != parameter.name) throw std::runtime_error("Checkpoint parameter name mismatch: " + parameter.name);
        if (reader.u32() != parameter.value.shape().size()) throw std::runtime_error("Checkpoint parameter rank mismatch: " + parameter.name);
        for (int dim : parameter.value.shape()) if (reader.u32() != static_cast<uint32_t>(dim)) throw std::runtime_error("Checkpoint parameter shape mismatch: " + parameter.name);
        const DType dtype = parameter.value.dtype();
        const bool low=is_low_precision_float(dtype);
        if ((!is_arithmetic_float(dtype)&&!is_complex(dtype)&&!low) || (version >= 2 && reader.u32() != uint32_t(dtype)) ||
            (version == 1 && dtype == DType::Float64) || (version<3&&low) || (version<4&&is_complex(dtype)))
            throw std::runtime_error("Checkpoint parameter dtype mismatch: " + parameter.name);
        Tensor proposed;
        if (version>=3) {
            parameters.push_back(reader.raw_values(dtype_storage_bytes(dtype,parameter.value.numel())));
            proposed=Tensor::from_raw(parameter.value.shape(),dtype,parameters.back(),parameter.value.backend());
        } else if (dtype==DType::Float64) {
            proposed=Tensor::from_double_data(parameter.value.shape(),reader.double_values(parameter.value.numel()),parameter.value.backend());
            parameters.push_back(proposed.raw_data());
        } else {
            proposed=Tensor::from_data(parameter.value.shape(),reader.values(parameter.value.numel()),parameter.value.backend(),false,dtype);
            parameters.push_back(proposed.raw_data());
        }
        finite_parameter(proposed);
        loaded.first_moment_float.emplace_back();loaded.second_moment_float.emplace_back();loaded.sgd_buffer_initialized.push_back(0);
        if (is_complex(dtype)) {
            const uint32_t initialized = reader.u32();
            if (initialized > 1 || (loaded.optimizer != "sgd" && initialized))
                throw std::runtime_error("Invalid complex checkpoint SGD history");
            loaded.sgd_buffer_initialized.back() = uint8_t(initialized);
            const size_t bytes = dtype_storage_bytes(dtype, parameter.value.numel());
            Tensor first = Tensor::from_raw(parameter.value.shape(), dtype, reader.raw_values(bytes), parameter.value.backend());
            Tensor second = Tensor::from_raw(parameter.value.shape(), dtype, reader.raw_values(bytes), parameter.value.backend());
            tensor_moment(first, parameter.value, dtype, false);
            tensor_moment(second, parameter.value, dtype, true);
            loaded.first_moment_double.push_back(std::move(first));
            loaded.second_moment_double.push_back(std::move(second));
            loaded.first_moment.emplace_back();
            loaded.second_moment.emplace_back();
        } else if (dtype == DType::Float64) {
            loaded.first_moment.emplace_back();
            loaded.second_moment.emplace_back();
            const std::vector<double> first = reader.double_values(parameter.value.numel()), second = reader.double_values(parameter.value.numel());
            for (double x : second) if (!(x >= 0.)) throw std::runtime_error("Invalid checkpoint second moment");
            loaded.first_moment_double.push_back(Tensor::from_double_data(parameter.value.shape(), first, parameter.value.backend()));
            loaded.second_moment_double.push_back(Tensor::from_double_data(parameter.value.shape(), second, parameter.value.backend()));
        } else if (low) {
            const uint32_t initialized=reader.u32();
            if (initialized>1 || (loaded.optimizer!="sgd"&&initialized)) throw std::runtime_error("Invalid checkpoint SGD history flag");
            loaded.sgd_buffer_initialized.back()=uint8_t(initialized);
            std::vector<float> first=reader.values(parameter.value.numel()),second=reader.values(parameter.value.numel());
            for(float x:second)if(x<0.f)throw std::runtime_error("Invalid checkpoint second moment");
            loaded.first_moment_float.back()=Tensor::from_data(parameter.value.shape(),first,parameter.value.backend());
            loaded.second_moment_float.back()=Tensor::from_data(parameter.value.shape(),second,parameter.value.backend());
            loaded.first_moment.emplace_back();loaded.second_moment.emplace_back();
            loaded.first_moment_double.emplace_back();loaded.second_moment_double.emplace_back();
        } else if (loaded.uses_resident_float32(parameter.value)) {
            const std::vector<float> first = reader.values(parameter.value.numel());
            const std::vector<float> second = reader.values(parameter.value.numel());
            for (float x : second) if (x < 0.f) throw std::runtime_error("Invalid checkpoint second moment");
            DeviceScope owner(parameter.value.backend(), parameter.value.device());
            loaded.first_moment_float.back() = Tensor::from_data(parameter.value.shape(), first, parameter.value.backend());
            loaded.second_moment_float.back() = Tensor::from_data(parameter.value.shape(), second, parameter.value.backend());
            loaded.first_moment.emplace_back();
            loaded.second_moment.emplace_back();
            loaded.first_moment_double.emplace_back();
            loaded.second_moment_double.emplace_back();
        } else {
            loaded.first_moment.push_back(reader.values(parameter.value.numel()));
            loaded.second_moment.push_back(reader.values(parameter.value.numel()));
            loaded.first_moment_double.emplace_back();
            loaded.second_moment_double.emplace_back();
        }
    }
    std::vector<Parameter> buffers = model.checkpoint_buffers();
    std::vector<std::vector<uint8_t>> buffer_values;
    if (has_buffers) {
        if (reader.u32() != buffers.size()) throw std::runtime_error("Checkpoint buffer count mismatch");
        for (const Parameter& buffer : buffers) {
            const Tensor& tensor = buffer.value;
            if (reader.string() != buffer.name || reader.u32() != uint32_t(tensor.dtype()) || reader.u32() != tensor.shape().size())
                throw std::runtime_error("Checkpoint buffer descriptor mismatch: " + buffer.name);
            for (int dim : tensor.shape()) if (reader.u32() != uint32_t(dim))
                throw std::runtime_error("Checkpoint buffer shape mismatch: " + buffer.name);
            buffer_values.push_back(reader.raw_values(dtype_storage_bytes(tensor.dtype(), tensor.numel())));
        }
    } else if (!buffers.empty()) {
        throw std::runtime_error("Checkpoint is missing persistent model buffers");
    }
    TrainingProgress loaded_progress;
    loaded_progress.attempted_steps = loaded.step;
    if (has_progress) {
        SamplerState& sampler = loaded_progress.sampler;
        loaded_progress.configuration = reader.string();
        const uint64_t count = reader.u64();
        const uint64_t offset = reader.u64();
        sampler.random_state = reader.u32();
        if (loaded_progress.configuration.empty() || offset >= count || !sampler.random_state ||
            (progress && count != expected_samples)) throw std::runtime_error("Invalid checkpoint sampler state");
        if (progress) {
            sampler.offset = static_cast<size_t>(offset);
            sampler.order.reserve(expected_samples);
        }
        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t index = reader.u64();
            if (index >= count) throw std::runtime_error("Invalid checkpoint sample index");
            if (progress) sampler.order.push_back(static_cast<size_t>(index));
        }
        if (envelope >= 6) {
            loaded_progress.attempted_steps = reader.u64();
            const uint32_t scale_bits = reader.u32();
            std::memcpy(&loaded_progress.loss_scale.scale, &scale_bits, sizeof(scale_bits));
            loaded_progress.loss_scale.finite_steps = reader.u32();
            if (loaded_progress.attempted_steps < loaded.step ||
                loaded_progress.attempted_steps > uint64_t(std::numeric_limits<int64_t>::max()) ||
                !std::isfinite(loaded_progress.loss_scale.scale) || loaded_progress.loss_scale.scale <= 0)
                throw std::runtime_error("Invalid checkpoint optimizer or loss scale progress");
        }
    }
    loaded_progress.checksum = reader.finish();
    for (size_t i = 0; i < parameters.size(); ++i) {
        Tensor& value = model.parameters()[i].value;
        value.assign_raw(parameters[i]);
    }
    for (size_t i = 0; i < buffer_values.size(); ++i) buffers[i].value.assign_raw(buffer_values[i]);
    if (state) *state = std::move(loaded);
    if (progress) *progress = std::move(loaded_progress);
}

}
