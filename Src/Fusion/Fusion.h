#pragma once

#include "Core/Backend.h"
#include <filesystem>

namespace elman::fusion {
struct Options {
    bool enabled = true;
    bool verify = true;
    std::filesystem::path cache_directory = "Intermediate/Fusion";
};
// Process-wide master switch. Thread-local policies can further disable fusion.
// Default: enabled. ELMAN_FUSION=0 disables it before the first use.
bool enabled();
void set_enabled(bool value);
struct Statistics {
    uint64_t compiled = 0, cache_hits = 0, launches = 0, fused_commands = 0;
    uint64_t verified = 0, rejected = 0, compile_failures = 0, barriers = 0;
    uint64_t elided_stores = 0;
};
// Policies and recording are local to the calling thread. Kernels keep FP32 storage.
class Policy {
public:
    explicit Policy(const Options& options);
    ~Policy() noexcept(false);
    Policy(const Policy&) = delete;
    Policy& operator=(const Policy&) = delete;
private:
    Options previous_;
};
// Explicit finish reports execution failures. Unwinding only runs the original commands.
// Publish tensors to other threads or native-pointer consumers only after finish().
// Framework reads and writes on this thread materialize the relevant pending work.
class Scope {
public:
    Scope();
    ~Scope() noexcept(false);
    void finish();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    bool active_ = false;
};
Statistics statistics();
void reset_statistics();
void flush();
// Internal GPU interception, after ordinary descriptor and dtype validation.
bool enqueue(const std::string& operation, const BufferPtr& output,
             const std::vector<BufferPtr>& inputs, const KernelParams& parameters);
// A read/write of unrelated storage does not force an otherwise safe region to end.
void access(const BufferPtr& buffer);
} // namespace elman::fusion
