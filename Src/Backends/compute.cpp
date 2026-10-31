#include "Core/backend.h"
#include "Backends/host_words.h"
#include "Backends/shader_source.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace elman {
namespace {
using Microsoft::WRL::ComPtr;

void d3d_check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        std::ostringstream message;
        message << operation << " failed (HRESULT 0x" << std::hex << static_cast<unsigned long>(result) << ')';
        throw std::runtime_error(message.str());
    }
}

struct ComputeState {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Buffer> constants;
    std::unordered_map<std::string, ComPtr<ID3D11ComputeShader>> shaders;
    std::mutex mutex;

    ComputeState() {
        const D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selected;
        d3d_check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            requested, 1, D3D11_SDK_VERSION, &device, &selected, &context), "D3D11CreateDevice");
        D3D11_BUFFER_DESC description{};
        description.ByteWidth = sizeof(KernelParams);
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        d3d_check(device->CreateBuffer(&description, nullptr, &constants), "Create constant buffer");
    }

    ID3D11ComputeShader* shader(const std::string& op) {
        auto found = shaders.find(op);
        if (found != shaders.end()) return found->second.Get();
        if (op.empty() || !std::all_of(op.begin(), op.end(), [](unsigned char ch) {
                return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_';
            })) throw std::runtime_error("Invalid Compute operator name");
        const auto path = runtime_shader_path("Compute", op + "_compute.hlsl");
        ComPtr<ID3DBlob> code;
        ComPtr<ID3DBlob> diagnostics;
        const std::string source = load_shader_source(path);
        const HRESULT result = D3DCompile(source.data(), source.size(), path.string().c_str(), nullptr, nullptr,
            "main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &diagnostics);
        if (FAILED(result)) {
            std::string detail;
            if (diagnostics) detail.assign(static_cast<const char*>(diagnostics->GetBufferPointer()), diagnostics->GetBufferSize());
            throw std::runtime_error("Compute shader compilation failed for " + op + ": " + detail);
        }
        ComPtr<ID3D11ComputeShader> compiled;
        d3d_check(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &compiled), "CreateComputeShader");
        auto inserted = shaders.emplace(op, std::move(compiled));
        return inserted.first->second.Get();
    }
};

ComputeState& compute_state() { static ComputeState state; return state; }

struct ComputeBuffer final : Buffer {
    ComPtr<ID3D11Buffer> data;
    ComPtr<ID3D11ShaderResourceView> read_view;
    ComPtr<ID3D11UnorderedAccessView> write_view;

    ComputeBuffer(std::size_t size, DType type) : Buffer(Backend::Compute, type, size) {
        const std::size_t words = std::max(size, std::size_t(1));
        if (words > std::numeric_limits<UINT>::max() / sizeof(std::uint32_t))
            throw std::runtime_error("Compute allocation exceeds the D3D11 buffer size limit");
        auto& state = compute_state();
        D3D11_BUFFER_DESC description{};
        description.ByteWidth = static_cast<UINT>(words * sizeof(std::uint32_t));
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        description.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        description.StructureByteStride = sizeof(std::uint32_t);
        std::vector<std::uint32_t> zeros(words);
        D3D11_SUBRESOURCE_DATA initial{};
        initial.pSysMem = zeros.data();
        d3d_check(state.device->CreateBuffer(&description, &initial, &data), "Create Compute buffer");
        D3D11_SHADER_RESOURCE_VIEW_DESC read_description{};
        read_description.Format = DXGI_FORMAT_UNKNOWN;
        read_description.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        read_description.Buffer.NumElements = static_cast<UINT>(words);
        d3d_check(state.device->CreateShaderResourceView(data.Get(), &read_description, &read_view), "Create Compute read view");
        D3D11_UNORDERED_ACCESS_VIEW_DESC write_description{};
        write_description.Format = DXGI_FORMAT_UNKNOWN;
        write_description.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        write_description.Buffer.NumElements = static_cast<UINT>(words);
        d3d_check(state.device->CreateUnorderedAccessView(data.Get(), &write_description, &write_view), "Create Compute write view");
    }
};

ComputeBuffer& checked_compute(const BufferPtr& buffer) {
    auto* value = dynamic_cast<ComputeBuffer*>(buffer.get());
    if (!value) throw std::runtime_error("Expected a Compute buffer");
    return *value;
}
}

bool available_compute() { try { compute_state(); return true; } catch (...) { return false; } }
BufferPtr allocate_compute(std::size_t count, DType dtype) { return std::make_shared<ComputeBuffer>(count, dtype); }

void upload_compute(const BufferPtr& buffer, const std::vector<float>& values) {
    auto& target = checked_compute(buffer);
    if (values.size() != target.count) throw std::runtime_error("Compute upload element count mismatch");
    if (values.empty()) return;
    const auto words = encode_words(values, target.dtype);
    auto& state = compute_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.context->UpdateSubresource(target.data.Get(), 0, nullptr, words.data(), 0, 0);
}

std::vector<float> download_compute(const BufferPtr& buffer) {
    auto& source = checked_compute(buffer);
    if (source.count == 0) return {};
    auto& state = compute_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    D3D11_BUFFER_DESC description{};
    source.data->GetDesc(&description);
    description.Usage = D3D11_USAGE_STAGING;
    description.BindFlags = 0;
    description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    description.MiscFlags = 0;
    description.StructureByteStride = 0;
    ComPtr<ID3D11Buffer> staging;
    d3d_check(state.device->CreateBuffer(&description, nullptr, &staging), "Create Compute readback buffer");
    state.context->CopyResource(staging.Get(), source.data.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};
    d3d_check(state.context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map Compute readback buffer");
    std::vector<std::uint32_t> words(source.count);
    std::memcpy(words.data(), mapped.pData, words.size() * sizeof(std::uint32_t));
    state.context->Unmap(staging.Get(), 0);
    return decode_words(words, source.dtype);
}

void dispatch_compute(const std::string& op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& params) {
    auto& target = checked_compute(output);
    if (inputs.size() > 3) throw std::runtime_error("Compute kernels accept at most three inputs");
    if (params.v[0] < 0 || static_cast<std::size_t>(params.v[0]) != target.count)
        throw std::runtime_error("Compute dispatch output count mismatch");
    if (target.count == 0) return;
    auto& state = compute_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ID3D11ShaderResourceView* source[3] = {nullptr, nullptr, nullptr};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) continue;
        auto& input = checked_compute(inputs[i]);
        if (input.data.Get() == target.data.Get()) throw std::runtime_error("Compute dispatch requires a distinct output buffer");
        source[i] = input.read_view.Get();
    }
    ID3D11ComputeShader* shader = state.shader(op);
    state.context->UpdateSubresource(state.constants.Get(), 0, nullptr, &params, 0, 0);
    ID3D11Buffer* constants = state.constants.Get();
    ID3D11UnorderedAccessView* destination = target.write_view.Get();
    state.context->CSSetShader(shader, nullptr, 0);
    state.context->CSSetConstantBuffers(0, 1, &constants);
    state.context->CSSetShaderResources(0, 3, source);
    state.context->CSSetUnorderedAccessViews(0, 1, &destination, nullptr);
    const UINT groups = static_cast<UINT>((target.count + 63) / 64);
    state.context->Dispatch(std::min(groups, UINT(65535)), (groups + 65534) / 65535, 1);
    ID3D11UnorderedAccessView* no_destination = nullptr;
    ID3D11ShaderResourceView* no_source[3] = {nullptr, nullptr, nullptr};
    state.context->CSSetUnorderedAccessViews(0, 1, &no_destination, nullptr);
    state.context->CSSetShaderResources(0, 3, no_source);
    d3d_check(state.device->GetDeviceRemovedReason(), "Compute dispatch");
}
}
