#pragma once

namespace elman {

EL_INLINE int broadcast_index(int flat, KernelInput input,
                              const KernelParams& params) {
  int result = 0;
  int delta = params.output_rank() - params.input_rank(input);
  for (int axis = params.output_rank() - 1; axis >= 0; --axis) {
    int coordinate = flat % params.output_shape(axis);
    flat /= params.output_shape(axis);
    int input_axis = axis - delta;
    if (input_axis >= 0 && params.input_shape(input, input_axis) != 1) {
      result += coordinate * params.input_stride(input, input_axis);
    }
  }
  return result;
}

EL_INLINE int input_count(KernelInput input, const KernelParams& params) {
  if (params.input_rank(input) < 0) {
    return 0;
  }
  int result = 1;
  for (int axis = 0; axis < params.input_rank(input); ++axis) {
    result *= params.input_shape(input, axis);
  }
  return result;
}

EL_INLINE int output_stride(int axis, const KernelParams& params) {
  int result = 1;
  for (int next = axis + 1; next < params.output_rank(); ++next) {
    result *= params.output_shape(next);
  }
  return result;
}

EL_INLINE int matrix_batch_offset(int flat, KernelInput input,
                                  const KernelParams& params) {
  int result = 0;
  int delta = params.output_rank() - params.input_rank(input);
  for (int axis = params.output_rank() - 3; axis >= 0; --axis) {
    int coordinate = flat % params.output_shape(axis);
    flat /= params.output_shape(axis);
    int input_axis = axis - delta;
    if (input_axis >= 0 && params.input_shape(input, input_axis) != 1) {
      result += coordinate * params.input_stride(input, input_axis);
    }
  }
  return result;
}

EL_INLINE float elman_random(int index, int seed, int offset) {
  EL_UINT value = EL_UINT(index) + EL_UINT(offset) * 747796405u +
                  EL_UINT(seed) * 2891336453u + 1u;
  value ^= value >> 16;
  value *= 2246822519u;
  value ^= value >> 13;
  value *= 3266489917u;
  value ^= value >> 16;
  return (float(value & 16777215u) + 0.5f) * (1.0f / 16777216.0f);
}

}  // namespace elman
