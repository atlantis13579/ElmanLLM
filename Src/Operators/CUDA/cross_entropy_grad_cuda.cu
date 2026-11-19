#include "kernel.cuh"
#include "row_reduce.cuh"

namespace elman {
__global__ void cross_entropy_gradient_rows(uint32_t* out, const uint32_t* a,
                                            const uint32_t* b,
                                            const uint32_t* c, KernelParams p) {
  __shared__ float scratch[256];
  int row = int(blockIdx.x),
      columns = p.input_shape(KernelInput::A, tensor_axis::logits::kClasses);
  float highest = -EL_INF;
  for (int column = int(threadIdx.x); column < columns;
       column += int(blockDim.x)) {
    highest = fmaxf(highest, READ_A(row * columns + column));
  }
  highest = row_reduce<true>(highest, scratch);
  float denominator = 0;
  for (int column = int(threadIdx.x); column < columns;
       column += int(blockDim.x)) {
    denominator += expf(READ_A(row * columns + column) - highest);
  }
  denominator = row_reduce<false>(denominator, scratch);
  int target = READ_INT_B(row);
  float scale = READ_C(0) / float(p.input_shape(KernelInput::A,
                                                tensor_axis::logits::kBatch));
  for (int column = int(threadIdx.x); column < columns;
       column += int(blockDim.x)) {
    int i = row * columns + column;
    WRITE_OUTPUT(
        i, (expf(READ_A(i) - highest) / denominator - float(column == target)) *
               scale);
  }
}
void launch_cross_entropy_grad_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    const KernelParams& p) {
  cross_entropy_gradient_rows<<<
      p.input_shape(KernelInput::A, tensor_axis::logits::kBatch), 256, 0, current_cuda_stream()>>>(
      out, a, b, c, p);
}
}  // namespace elman
