#pragma once
#include "Core/Backend.h"
#include <functional>
#include <memory>

namespace elman {
// Records already validated native CUDA dispatches and their scratch buffers.
// Capture reuses the exact storage and parameters; callers validate all mutable
// input domains before replay. No host computation is included in the graph.
class CommandGraph {
public:
    CommandGraph();
    ~CommandGraph();
    CommandGraph(const CommandGraph&) = delete;
    CommandGraph& operator=(const CommandGraph&) = delete;
    void record(const std::function<void()>& forward);
    void compile();
    void replay();
    void synchronize() const;
    size_t command_count() const;
    size_t node_count() const;
    size_t retained_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#ifdef ELMAN_HAS_CUDA
namespace detail {
class CudaCommandRecording {
public:
    CudaCommandRecording(const std::string&, const BufferPtr&, const std::vector<BufferPtr>&, const KernelParams&);
    ~CudaCommandRecording();
private:
    void* previous_ = nullptr;
};
bool lowering_cuda_graph();
bool recording_cuda_graph();
BufferPtr replay_cuda_graph_workspace(size_t bytes);
void record_cuda_graph_workspace(const BufferPtr& buffer);
}
#endif
} // namespace elman
