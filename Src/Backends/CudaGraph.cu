#include "Core/CommandGraph.h"
#include "Core/Device.h"
#include "Core/Memory.h"
#include "Backends/CudaExecution.h"
#include <set>
#include <stdexcept>

namespace elman {
void dispatch_cuda(const std::string&, const BufferPtr&, const std::vector<BufferPtr>&, const KernelParams&);
void prepare_cuda_graph_buffers(const std::vector<BufferPtr>&);
void finish_cuda_graph_buffers(const std::vector<BufferPtr>&);
namespace {
struct Command {
    std::string operation;
    BufferPtr output;
    std::vector<BufferPtr> inputs, workspace;
    KernelParams parameters;
};
thread_local std::vector<std::unique_ptr<Command>>* recording = nullptr;
thread_local Command* current_command = nullptr;
thread_local bool lowering = false;
thread_local size_t workspace_index = 0;
void require(bool valid, const char* message) { if (!valid) throw std::runtime_error(message); }
struct LoweringScope {
    LoweringScope() { require(!lowering && !recording, "Nested CUDA graph lowering"); lowering = true; }
    ~LoweringScope() { lowering = false; current_command = nullptr; }
};
}
namespace detail {
CudaCommandRecording::CudaCommandRecording(const std::string& op, const BufferPtr& output,
    const std::vector<BufferPtr>& inputs, const KernelParams& parameters) : previous_(current_command) {
    if (!recording) return;
    require(!current_command, "Nested CUDA dispatch cannot be recorded as an independent command");
    auto command = std::make_unique<Command>();
    command->operation = op; command->output = output; command->inputs = inputs; command->parameters = parameters;
    current_command = command.get();
    recording->push_back(std::move(command));
}
CudaCommandRecording::~CudaCommandRecording() { current_command = static_cast<Command*>(previous_); }
bool lowering_cuda_graph() { return lowering; }
BufferPtr replay_cuda_graph_workspace(size_t bytes) {
    if (!lowering) return {};
    require(current_command && workspace_index < current_command->workspace.size(), "CUDA graph scratch count changed");
    auto buffer = current_command->workspace[workspace_index++];
    require(buffer->bytes == bytes, "CUDA graph scratch size changed");
    return buffer;
}
void record_cuda_graph_workspace(const BufferPtr& buffer) {
    if (recording && current_command) current_command->workspace.push_back(buffer);
}
}
struct CommandGraph::Impl {
    int device = current_device(Backend::CUDA);
    ExecutionStream stream{Backend::CUDA};
    std::vector<std::unique_ptr<Command>> commands;
    std::vector<BufferPtr> buffers;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    size_t nodes = 0, bytes = 0;
    bool recorded = false, failed = false;
    ~Impl() {
        DeviceScope owner(Backend::CUDA, device);
        try { stream.synchronize(); } catch (...) {}
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
CommandGraph::CommandGraph() : impl_(std::make_unique<Impl>()) {}
CommandGraph::~CommandGraph() = default;
void CommandGraph::record(const std::function<void()>& forward) {
    auto& s = *impl_;
    require(!s.recorded && !s.failed && !recording && !lowering, "CUDA graph recording is not reusable or nestable");
    DeviceScope owner(Backend::CUDA, s.device);
    StreamScope scope(s.stream, true, false);
    recording = &s.commands;
    try { forward(); s.stream.synchronize(); }
    catch (...) { recording = nullptr; current_command = nullptr; s.failed = true; throw; }
    recording = nullptr;
    require(!s.commands.empty(), "CUDA graph recording contains no native commands");
    std::set<const Buffer*> seen;
    auto retain = [&](const BufferPtr& buffer) {
        if (!buffer) return;
        require(buffer->backend == Backend::CUDA && buffer->device_ordinal() == s.device, "CUDA graph has mixed devices");
        if (seen.insert(&buffer->allocation()).second) {
            s.buffers.push_back(buffer);
            s.bytes = checked_memory_add(s.bytes, buffer->allocation().bytes);
        }
    };
    for (const auto& command : s.commands) {
        retain(command->output);
        for (const auto& input : command->inputs) retain(input);
        for (const auto& scratch : command->workspace) retain(scratch);
    }
    s.recorded = true;
}
void CommandGraph::compile() {
    auto& s = *impl_;
    require(s.recorded && !s.failed && !s.executable, "CUDA graph is not ready for capture");
    DeviceScope owner(Backend::CUDA, s.device);
    StreamScope scope(s.stream, true, false);
    std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
    s.stream.synchronize();
    prepare_cuda_graph_buffers(s.buffers);
    s.stream.synchronize();
    LoweringScope lowering_scope;
    check_cuda_execution(cudaStreamBeginCapture(current_cuda_stream(), cudaStreamCaptureModeThreadLocal), "Begin whole-model capture");
    try {
        for (const auto& command : s.commands) {
            current_command = command.get(); workspace_index = 0;
            dispatch_cuda(command->operation, command->output, command->inputs, command->parameters);
            require(workspace_index == command->workspace.size(), "CUDA graph did not consume all recorded scratch");
        }
    } catch (...) {
        cudaStreamEndCapture(current_cuda_stream(), &s.graph);
        cudaGetLastError(); s.failed = true; throw;
    }
    current_command = nullptr;
    try {
        check_cuda_execution(cudaStreamEndCapture(current_cuda_stream(), &s.graph), "End whole-model capture");
        check_cuda_execution(cudaGraphInstantiate(&s.executable, s.graph, 0), "Instantiate whole-model graph");
        check_cuda_execution(cudaGraphGetNodes(s.graph, nullptr, &s.nodes), "Count whole-model graph nodes");
        require(s.nodes > 0, "Whole-model capture is empty");
    } catch (...) { s.failed = true; throw; }
}
void CommandGraph::replay() {
    auto& s = *impl_;
    require(s.executable && !s.failed, "CUDA graph is not executable");
    DeviceScope owner(Backend::CUDA, s.device);
    StreamScope scope(s.stream, true, false);
    std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
    prepare_cuda_graph_buffers(s.buffers);
    try {
        check_cuda_execution(cudaGraphLaunch(s.executable, current_cuda_stream()), "Replay whole-model graph");
        finish_cuda_graph_buffers(s.buffers);
    } catch (...) { s.failed = true; try { s.stream.synchronize(); } catch (...) {} throw; }
}
void CommandGraph::synchronize() const { impl_->stream.synchronize(); }
size_t CommandGraph::command_count() const { return impl_->commands.size(); }
size_t CommandGraph::node_count() const { return impl_->nodes; }
size_t CommandGraph::retained_bytes() const { return impl_->bytes; }
} // namespace elman
