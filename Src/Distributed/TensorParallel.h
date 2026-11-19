#pragma once

#include "Core/Tensor.h"
#include "Distributed/ProcessGroup.h"

namespace elman {

// A column-parallel projection stores its own output rows. A row-parallel
// projection reduces the partial outputs from its local input columns.
Tensor reduce_tensor_parallel(const Tensor& local, ProcessGroup& group);

struct TensorParallelHeads {
  int query_offset, query_heads;
  int kv_offset, kv_heads;
};

TensorParallelHeads partition_attention_heads(int heads, int kv_heads, int rank, int size);

}  // namespace elman
