#include "Distributed/TensorParallel.h"

#include <stdexcept>

#include "Core/Device.h"
#include "Core/Graph.h"

namespace elman {

Tensor reduce_tensor_parallel(const Tensor& local, ProcessGroup& group) {
  if (group.size() == 1) return local;
  if (!is_compute_float(local.dtype())) throw std::invalid_argument("Tensor parallel reduction requires a compute dtype");
  DeviceScope owner(local.backend(), local.device());
  auto values = local.data();
  group.all_reduce_sum(values.data(), values.size());
  return Tensor::from_data(local.shape(), values, local.backend(), false, local.dtype());
}

TensorParallelHeads partition_attention_heads(int heads, int kv_heads, int rank, int size) {
  if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || size <= 0 || rank < 0 || rank >= size ||
      heads % size || (kv_heads % size && size % kv_heads))
    throw std::invalid_argument("Tensor parallel ranks must partition query heads and partition or replicate KV heads");
  const int queries = heads / size;
  if (kv_heads >= size) return {rank * queries, queries, rank * (kv_heads / size), kv_heads / size};
  return {rank * queries, queries, rank / (size / kv_heads), 1};
}

}  // namespace elman
