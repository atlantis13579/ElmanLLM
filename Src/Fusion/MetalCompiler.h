#pragma once
#include "Fusion/Compiler.h"

namespace elman::fusion {
std::shared_ptr<Executable> compile_metal(const Graph& graph, const Options& options, Statistics& stats);
} // namespace elman::fusion
