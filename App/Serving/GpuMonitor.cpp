#include "GpuMonitor.h"
#include "ServerLog.h"

#ifdef ELMAN_HAS_NVML
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <filesystem>
#else
#include <dlfcn.h>
#endif

#include <cuda_runtime_api.h>
#include <nvml.h>

#include "Core/Device.h"

#include <chrono>
#include <condition_variable>
#include <sstream>
#include <thread>
#include <utility>
#endif

namespace elman {

#ifdef ELMAN_HAS_NVML
namespace {

#ifdef _WIN32
using Library = HMODULE;

Library open_nvml() {
    if (HMODULE library = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) return library;
    wchar_t program_files[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"ProgramW6432", program_files, MAX_PATH);
    if (!length || length >= MAX_PATH) return nullptr;
    const std::filesystem::path path = std::filesystem::path(program_files) / L"NVIDIA Corporation" / L"NVSMI" / L"nvml.dll";
    if (!path.is_absolute()) return nullptr;
    return LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
}

void close_library(Library library) { FreeLibrary(library); }
FARPROC symbol(Library library, const char* name) { return GetProcAddress(library, name); }
#else
using Library = void*;

Library open_nvml() { return dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL); }
void close_library(Library library) { dlclose(library); }
void* symbol(Library library, const char* name) { return dlsym(library, name); }
#endif

template<class Function>
Function load_function(Library library, const char* name) {
    return reinterpret_cast<Function>(symbol(library, name));
}

} // namespace

struct GpuMonitor::Impl {
    Library library = nullptr;
    decltype(&nvmlShutdown) shutdown = nullptr;
    decltype(&nvmlDeviceGetUtilizationRates) utilization = nullptr;
    decltype(&nvmlDeviceGetMemoryInfo) memory = nullptr;
    nvmlDevice_t device = nullptr;
    bool initialized = false;
    std::string label;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false;

    bool start(int ordinal, std::function<bool()> has_active_task) {
        char bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), ordinal) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        library = open_nvml();
        if (!library) return false;
        const decltype(&nvmlInit_v2) init = load_function<decltype(&nvmlInit_v2)>(library, "nvmlInit_v2");
        const decltype(&nvmlDeviceGetHandleByPciBusId_v2) resolve = load_function<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>(library, "nvmlDeviceGetHandleByPciBusId_v2");
        shutdown = load_function<decltype(shutdown)>(library, "nvmlShutdown");
        utilization = load_function<decltype(utilization)>(library, "nvmlDeviceGetUtilizationRates");
        memory = load_function<decltype(memory)>(library, "nvmlDeviceGetMemoryInfo");
        if (!init || !resolve || !shutdown || !utilization || !memory || init() != NVML_SUCCESS) return false;
        initialized = true;
        if (resolve(bus_id, &device) != NVML_SUCCESS) return false;
        label = "gpu " + std::to_string(ordinal);
        worker = std::thread([this, has_active_task = std::move(has_active_task)] {
            std::unique_lock<std::mutex> lock(mutex);
            while (!changed.wait_for(lock, std::chrono::seconds(5), [this] { return stopping; })) {
                lock.unlock();
                try { if (has_active_task()) report(); } catch (...) { return; }
                lock.lock();
            }
        });
        return true;
    }

    ~Impl() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            changed.notify_one();
            worker.join();
        }
        if (initialized) shutdown();
        if (library) close_library(library);
    }

    void report() const {
        nvmlUtilization_t rates{};
        nvmlMemory_t vram{};
        constexpr unsigned long long mib = 1024ULL * 1024;
        std::ostringstream output;
        output << "utilization ";
        if (utilization(device, &rates) == NVML_SUCCESS) output << rates.gpu << '%';
        else output << "unavailable";
        output << " | VRAM ";
        if (memory(device, &vram) == NVML_SUCCESS) output << vram.used / mib << " / " << vram.total / mib << " MiB";
        else output << "unavailable";
        server_status(label, output.str());
    }
};
#else
struct GpuMonitor::Impl {};
#endif

GpuMonitor::GpuMonitor(Backend backend, std::function<bool()> has_active_task) {
    if (backend != Backend::CUDA) return;
#ifdef ELMAN_HAS_NVML
    try {
        std::unique_ptr<GpuMonitor::Impl> monitor = std::make_unique<Impl>();
        if (monitor->start(current_device(backend), std::move(has_active_task))) {
            impl_ = std::move(monitor);
            return;
        }
    } catch (const std::exception&) {}
#endif
    server_status("gpu", "NVIDIA GPU telemetry unavailable");
}

GpuMonitor::~GpuMonitor() = default;

} // namespace elman
