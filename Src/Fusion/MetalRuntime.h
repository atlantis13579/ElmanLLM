#pragma once
#include "Core/Backend.h"
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace elman::fusion {
// The backend owns the queue, buffer views, and synchronization. Compilers never
// create a second queue or access shared storage before preceding work completes.
MPSGraphDevice* metal_graph_device();
void encode_metal(MPSGraphExecutable* executable, const std::vector<BufferPtr>& inputs,
                  const std::vector<BufferPtr>& outputs);
} // namespace elman::fusion
