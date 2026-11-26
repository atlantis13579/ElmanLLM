#pragma once
#include "Fusion/Fusion.h"
#include "Fusion/IR.h"
#include <memory>

namespace elman::fusion {
struct Executable {
    virtual ~Executable() = default;
    virtual void launch(const Graph& graph, const Schedule& schedule) = 0;
    bool verified = false, rejected = false;
    std::string diagnostic;
};
} // namespace elman::fusion
