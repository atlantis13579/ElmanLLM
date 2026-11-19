#include "kernel.cuh"

namespace elman {
namespace {
struct RepeatGradientShape {
  int rank;
  int input_stride[8]{};
  int period[8]{};
  int replicas[8]{};
  int replica_count = 1;

  explicit RepeatGradientShape(const KernelParams& p) : rank(p.input_rank(KernelInput::A)) {
    const int leading = rank - p.output_rank();
    for (int axis = 0; axis < rank; ++axis) {
      input_stride[axis] = p.input_stride(KernelInput::A, axis);
      period[axis] = axis < leading ? 1 : p.output_shape(axis - leading);
      replicas[axis] = p.input_shape(KernelInput::A, axis) / period[axis];
      replica_count *= replicas[axis];
    }
  }
};

__global__ void repeat_gradient(uint32_t* out, const uint32_t* a,
                                KernelParams p, RepeatGradientShape shape) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(p.output_count())) return;
  int flat = i, first = 0;
  for (int axis = shape.rank - 1; axis >= 0; --axis) {
    first += (flat % shape.period[axis]) * shape.input_stride[axis];
    flat /= shape.period[axis];
  }
  float value = 0;
  // Enumerate only replicas of this element, in input order.
  for (int replica = 0; replica < shape.replica_count; ++replica) {
    int coordinate = replica, source = first;
    for (int axis = shape.rank - 1; axis >= 0; --axis) {
      source += (coordinate % shape.replicas[axis]) * shape.period[axis] * shape.input_stride[axis];
      coordinate /= shape.replicas[axis];
    }
    value += READ_A(source);
  }
  WRITE_OUTPUT(i, value);
}
}  // namespace

void launch_repeat_grad_cuda(uint32_t* out, const uint32_t* a, const uint32_t*,
                             const uint32_t*, const KernelParams& p) {
  if (p.output_count())
    repeat_gradient<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0,
                      current_cuda_stream()>>>(out, a, p, RepeatGradientShape(p));
}

void launch_repeat_backward_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                 const uint32_t* c, const KernelParams& p) {
  launch_repeat_grad_cuda(out, a, b, c, p);
}
}  // namespace elman
