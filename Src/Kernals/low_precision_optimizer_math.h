#pragma once
#include "Kernals/float_optimizer_math.h"
#include "Kernals/low_precision_optimizer.h"
#include "Kernals/low_precision_pointwise_math.h"

namespace elman {
inline FloatOptimizerParameters float_optimizer_parameters(const KernelParams& p) {
    // Only the low-format descriptor stores the square root of the second bias correction.
    const double correction2 = optimizer_parameter(p, 8);
    return {optimizer_parameter(p, 0), optimizer_parameter(p, 1), optimizer_parameter(p, 2),
            optimizer_parameter(p, 3), optimizer_parameter(p, 4), optimizer_parameter(p, 5),
            optimizer_parameter(p, 6), optimizer_parameter(p, 7),
            p.attribute(optimizer_attr::kResidentFloat32) ? std::sqrt(correction2) : correction2};
}

#ifdef __CUDACC__
__device__
#endif
inline float optimizer_result(int index, const uint32_t* weights, const uint32_t* gradients,
                              const uint32_t* first, const uint32_t* second, const KernelParams& p,
                              const FloatOptimizerParameters& scalars) {
    const DType dtype = DType(p.input_dtype(KernelInput::A));
    const FloatOptimizerElement result = float_optimizer_element(
        low_precision_value(weights, index, dtype), low_precision_value(gradients, index, dtype),
        reinterpret_cast<const float*>(first)[index], reinterpret_cast<const float*>(second)[index],
        scalars, static_cast<FloatOptimizerAlgorithm>(p.attribute(optimizer_attr::kAlgorithm)), p.attribute(optimizer_attr::kFirstStep) != 0,
        index, p.output_count());
    const int mode = p.attribute(optimizer_attr::kResult);
    return mode == 1 ? result.first : mode == 2 ? result.second : result.value;
}
} // namespace elman
