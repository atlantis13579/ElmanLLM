#include "Core/Memory.h"
#include "Core/Device.h"
#ifdef ELMAN_HAS_METAL
#include "Backends/Metal.h"
#endif

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace elman {
void prepare_cached_allocation(Backend backend, size_t bytes);
size_t reclaim_cached_allocations(Backend backend);
#ifdef ELMAN_HAS_CUDA
MemoryInfo memory_info_cuda();
#endif
namespace {
struct LimitEntry { size_t ceiling, peak; };
struct Account { MemoryStats stats; std::map<uint64_t, LimitEntry> limits; };
using AccountKey = std::pair<Backend, int>;
struct Ledger { std::mutex mutex; std::map<AccountKey, Account> accounts; uint64_t next = 0; };
Ledger& ledger() {
    // Outlive thread-local vendor contexts whose workspaces release at exit.
    static Ledger* value = new Ledger;
    return *value;
}
void validate_backend(Backend backend) {
    if (backend != Backend::CPU && backend != Backend::CUDA && backend != Backend::Metal)
        throw std::invalid_argument("Invalid memory backend");
}
size_t& category(MemoryStats& stats, MemoryKind kind) {
    if (kind == MemoryKind::Cached) return stats.cached_bytes;
    return kind == MemoryKind::Tensor ? stats.tensor_bytes : stats.workspace_bytes;
}
}
MemoryLimitExceeded::MemoryLimitExceeded(Backend backend, size_t requested, size_t live, size_t limit)
    : std::runtime_error("Memory limit exceeded on " + backend_name(backend) + ": requested=" +
        std::to_string(requested) + ", live=" + std::to_string(live) + ", limit=" + std::to_string(limit) + " bytes") {}
size_t checked_memory_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::overflow_error("Memory estimate addition overflow");
    return a + b;
}
size_t checked_memory_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) throw std::overflow_error("Memory estimate multiplication overflow");
    return a * b;
}
size_t memory_allocation_bytes(Backend backend, size_t payload) {
    validate_backend(backend);
    return backend == Backend::CPU ? checked_memory_add(payload, 7) / 8 * 8 : std::max(payload, size_t(1));
}
MemoryStats memory_stats(Backend backend) {
#ifndef ELMAN_HAS_CUDA
    if (backend == Backend::CUDA) return {};
#endif
#ifndef ELMAN_HAS_METAL
    if (backend == Backend::Metal) return {};
#endif
    return memory_stats(backend, current_device(backend));
}
MemoryStats memory_stats(Backend backend, int device) {
    validate_backend(backend);
    if ((backend == Backend::CPU && device != -1) || (backend == Backend::CUDA && device < 0) ||
        (backend == Backend::Metal && device != 0))
        throw std::invalid_argument("Invalid memory account device");
    Ledger& value = ledger();
    std::lock_guard<std::mutex> lock(value.mutex);
    return value.accounts[{backend, device}].stats;
}
MemoryReservation::MemoryReservation(Backend backend, size_t bytes, MemoryKind kind)
    : backend_(backend), device_(current_device(backend)), kind_(kind) {
    prepare_cached_allocation(backend, bytes);
    Ledger& value = ledger();
    std::unique_lock<std::mutex> lock(value.mutex);
    Account& account = value.accounts[{backend, device_}];
    size_t next = 0;
    for (int attempt = 0; ; ++attempt) {
        size_t current = account.stats.live_bytes();
        next = checked_memory_add(current, bytes);
        size_t ceiling = std::numeric_limits<size_t>::max();
        for (const std::pair<const uint64_t, LimitEntry>& entry : account.limits) ceiling = std::min(ceiling, entry.second.ceiling);
        if (next <= ceiling) break;
        if (attempt) throw MemoryLimitExceeded(backend, bytes, current, ceiling);
        lock.unlock();
        size_t reclaimed = reclaim_cached_allocations(backend);
        lock.lock();
        if (!reclaimed) throw MemoryLimitExceeded(backend, bytes, current, ceiling);
    }
    category(account.stats, kind) += bytes;
    account.stats.peak_bytes = std::max(account.stats.peak_bytes, next);
    ++account.stats.allocations;
    for (std::pair<const uint64_t, LimitEntry>& entry : account.limits) entry.second.peak = std::max(entry.second.peak, next);
    bytes_ = bytes;
}
void MemoryReservation::release() noexcept {
    if (!bytes_) return;
    Ledger& value = ledger();
    std::lock_guard<std::mutex> lock(value.mutex);
    category(value.accounts[{backend_, device_}].stats, kind_) -= bytes_;
    bytes_ = 0;
}
MemoryReservation::~MemoryReservation() { release(); }
MemoryReservation::MemoryReservation(MemoryReservation&& other) noexcept
    : backend_(other.backend_), device_(other.device_), kind_(other.kind_), bytes_(std::exchange(other.bytes_, 0)) {}
MemoryReservation& MemoryReservation::operator=(MemoryReservation&& other) noexcept {
    if (this != &other) {
        release(); backend_ = other.backend_; device_ = other.device_; kind_ = other.kind_; bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}
void MemoryReservation::reclassify(MemoryKind kind) {
    if (kind == kind_) return;
    Ledger& value = ledger();
    std::lock_guard<std::mutex> lock(value.mutex);
    MemoryStats& stats = value.accounts[{backend_, device_}].stats;
    category(stats, kind_) -= bytes_;
    category(stats, kind) += bytes_;
    kind_ = kind;
}
MemoryLimit::MemoryLimit(Backend backend, size_t bytes) : backend_(backend), device_(current_device(backend)) {
    Ledger& value = ledger();
    std::lock_guard<std::mutex> lock(value.mutex);
    Account& account = value.accounts[{backend, device_}];
    size_t current = account.stats.live_bytes();
    if (current > bytes) throw MemoryLimitExceeded(backend, 0, current, bytes);
    id_ = ++value.next;
    account.limits.emplace(id_, LimitEntry{bytes, current});
}
MemoryLimit::~MemoryLimit() {
    Ledger& value = ledger();
    std::lock_guard<std::mutex> lock(value.mutex);
    value.accounts[{backend_, device_}].limits.erase(id_);
}
size_t MemoryLimit::peak_bytes() const {
    Ledger& value = ledger();
    std::lock_guard<std::mutex> lock(value.mutex);
    return value.accounts[{backend_, device_}].limits.at(id_).peak;
}
MemoryInfo memory_info(Backend backend) {
    validate_backend(backend);
    if (backend == Backend::Metal) {
#ifdef ELMAN_HAS_METAL
        return memory_info_metal();
#else
        throw std::runtime_error("Metal memory information unavailable");
#endif
    }
    if (backend == Backend::CUDA) {
#ifdef ELMAN_HAS_CUDA
        return memory_info_cuda();
#else
        throw std::runtime_error("CUDA memory information unavailable");
#endif
    }
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) throw std::runtime_error("Cannot query host memory");
    return {size_t(status.ullTotalPhys), size_t(std::min(status.ullAvailPhys, status.ullAvailPageFile))};
#elif defined(__APPLE__)
    uint64_t total = 0;
    size_t length = sizeof(total);
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_size = 0;
    const mach_port_t host = mach_host_self();
    const kern_return_t queried = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics), &count);
    const kern_return_t page_status = host_page_size(host, &page_size);
    mach_port_deallocate(mach_task_self(), host);
    if (sysctlbyname("hw.memsize", &total, &length, nullptr, 0) || queried != KERN_SUCCESS || page_status != KERN_SUCCESS)
        throw std::runtime_error("Cannot query macOS host memory");
    const size_t pages = size_t(statistics.free_count) + statistics.inactive_count + statistics.purgeable_count;
    return {size_t(total), std::min(size_t(total), checked_memory_mul(pages, page_size))};
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES), available = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    if (pages < 0 || available < 0 || page_size <= 0) throw std::runtime_error("Cannot query host memory");
    MemoryInfo result{checked_memory_mul(size_t(pages), size_t(page_size)), checked_memory_mul(size_t(available), size_t(page_size))};
    std::ifstream file("/proc/meminfo");
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream input(line);
        std::string name; size_t kib;
        if (input >> name >> kib && name == "MemAvailable:") result.available_bytes = checked_memory_mul(kib, 1024);
    }
    return result;
#else
    throw std::runtime_error("Host memory information is unavailable on this platform");
#endif
}
size_t memory_available_limit(Backend backend, size_t maximum, size_t reserve) {
    const int device = current_device(backend);
    MemoryInfo info = memory_info(backend);
    size_t free = info.available_bytes > reserve ? info.available_bytes - reserve : 0;
    size_t limit = checked_memory_add(memory_stats(backend).live_bytes(), free);
    if (maximum) limit = std::min(maximum, limit);
    Ledger& value = ledger();
    std::lock_guard<std::mutex> lock(value.mutex);
    for (const std::pair<const uint64_t, LimitEntry>& entry : value.accounts[{backend, device}].limits) limit = std::min(limit, entry.second.ceiling);
    return limit;
}
} // namespace elman
