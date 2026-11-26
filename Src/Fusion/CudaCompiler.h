#pragma once
#include "Fusion/Compiler.h"

namespace elman::fusion {
std::shared_ptr<Executable> compile(const Graph& graph, const Schedule& schedule, const Options& options, Statistics& stats);
void submit(const Graph& graph, Executable& executable, const Schedule& schedule);
} // namespace elman::fusion
