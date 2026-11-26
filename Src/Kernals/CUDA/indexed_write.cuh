#pragma once

#include "kernel.cuh"

namespace elman {
enum class IndexedWrite { GatherGradient, Scatter, ScatterAdd };

template <IndexedWrite operation>
__global__ void indexed_write(uint32_t* out, const uint32_t* a, const uint32_t* b,
                              const uint32_t* c, KernelParams p, int axis, int inner) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(p.output_count())) return;
  constexpr bool gather = operation == IndexedWrite::GatherGradient;
  constexpr bool replace = operation == IndexedWrite::Scatter;
  constexpr KernelInput source = gather ? KernelInput::A : KernelInput::C;
  const int selected = (i / inner) % p.output_shape(axis);
  int flat = i, base = 0;
  int length = p.input_shape(source, axis);
  for (int dim = p.output_rank() - 1; dim >= 0; --dim) {
    const int extent = p.output_shape(dim);
    const int coordinate = flat % extent;
    flat /= extent;
    if (dim != axis) {
      if (coordinate >= p.input_shape(source, dim)) {
        length = 0;
        break;
      }
      base += coordinate * p.input_stride(source, dim);
    }
  }
  const int stride = p.input_stride(source, axis);
  if constexpr (replace) {
    // The last source occurrence wins, even when the index repeats.
    for (int k = length - 1; k >= 0; --k) {
      const int index = base + k * stride;
      if (READ_INT_B(index) == selected) {
        if (p.input_dtype(KernelInput::C) == p.output_dtype())
          COPY_TYPED_ELEMENT(out, c, i, index, p.output_dtype());
        else if (kernel_integer(p.output_dtype())) WRITE_INT_OUTPUT(i, READ_INT_C(index));
        else WRITE_OUTPUT(i, READ_C(index));
        return;
      }
    }
    COPY_INPUT_A(i, i);
  } else {
    const bool integral = !gather && kernel_integer(p.output_dtype());
    uint32_t integer = integral ? uint32_t(READ_INT_A(i)) : 0;
    float value = gather || integral ? 0.f : READ_A(i);
    for (int k = 0; k < length; ++k) {
      const int index = base + k * stride;
      if (READ_INT_B(index) != selected) continue;
      if constexpr (gather) value += READ_A(index);
      else if (integral) integer += uint32_t(READ_INT_C(index));
      else value += READ_C(index);
    }
    if (integral) WRITE_INT_OUTPUT(i, int(integer));
    else WRITE_OUTPUT(i, value);
  }
}

template <IndexedWrite operation>
void launch_indexed_write(uint32_t* out, const uint32_t* a, const uint32_t* b,
                          const uint32_t* c, const KernelParams& p) {
  if (p.output_count() <= 0) return;
  const int axis = p.attribute(operator_attr::indexing::kAxis);
  int inner = 1;
  for (int dim = axis + 1; dim < p.output_rank(); ++dim) inner *= p.output_shape(dim);
  indexed_write<operation><<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0,
                             current_cuda_stream()>>>(out, a, b, c, p, axis, inner);
}
}  // namespace elman
