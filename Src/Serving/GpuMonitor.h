#pragma once

#include "Core/Backend.h"

#include <functional>
#include <memory>

namespace elman {

// Samples the CUDA device during active inference.
class GpuMonitor {
public:
    GpuMonitor(Backend backend, std::function<bool()> has_active_task);
    ~GpuMonitor();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace elman
