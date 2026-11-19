#include "Core/CommandGraph.h"
#include <stdexcept>
#ifndef ELMAN_HAS_CUDA
namespace elman {
struct CommandGraph::Impl {};
CommandGraph::CommandGraph() { throw std::runtime_error("CUDA Graph requires a CUDA build"); }
CommandGraph::~CommandGraph() = default;
void CommandGraph::record(const std::function<void()>&) { throw std::runtime_error("CUDA Graph unavailable"); }
void CommandGraph::compile() { throw std::runtime_error("CUDA Graph unavailable"); }
void CommandGraph::replay() { throw std::runtime_error("CUDA Graph unavailable"); }
void CommandGraph::synchronize() const {}
size_t CommandGraph::command_count() const { return 0; }
size_t CommandGraph::node_count() const { return 0; }
size_t CommandGraph::retained_bytes() const { return 0; }
}
#endif
