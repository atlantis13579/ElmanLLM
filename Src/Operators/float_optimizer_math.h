#pragma once

#include <cmath>
#include <cstddef>

namespace elman {

// Derive scalar coefficients once, before rounding them to the tensor's computation dtype.
struct FloatOptimizerParameters {
    FloatOptimizerParameters(double rate, double scale, double decay, double momentum,
                             double beta1, double beta2, double epsilon,
                             double correction1, double correction2_sqrt)
        : rate(float(rate)), scale(float(scale)), decay(float(decay)), momentum(float(momentum)),
          mean_weight(float(1. - beta1)), beta2(float(beta2)), square_weight(float(1. - beta2)),
          epsilon(float(epsilon)), correction2_sqrt(float(correction2_sqrt)),
          decay_factor(float(1. - rate * decay)), step_size(float(rate / correction1)) {}

    float rate, scale, decay, momentum;
    float mean_weight, beta2, square_weight, epsilon, correction2_sqrt, decay_factor, step_size;
};

enum class FloatOptimizerAlgorithm { Sgd, Adam, AdamW };

struct FloatOptimizerElement {
    float value, first, second;
};

#ifdef __CUDACC__
#define ELMAN_OPTIMIZER_MATH __device__
#else
#define ELMAN_OPTIMIZER_MATH
#endif

ELMAN_OPTIMIZER_MATH inline float optimizer_mul(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    volatile float result = a * b;
    return result;
#endif
}
ELMAN_OPTIMIZER_MATH inline float optimizer_add(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    volatile float result = a + b;
    return result;
#endif
}
ELMAN_OPTIMIZER_MATH inline float optimizer_sub(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fsub_rn(a, b);
#else
    volatile float result = a - b;
    return result;
#endif
}
ELMAN_OPTIMIZER_MATH inline float optimizer_div(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a, b);
#else
    volatile float result = a / b;
    return result;
#endif
}
ELMAN_OPTIMIZER_MATH inline float optimizer_scaled_add(float a, float b, float scale, size_t index, size_t count) {
#ifdef __CUDA_ARCH__
    return ::fmaf(scale, b, a);
#else
    // CPU TensorIterator uses two AVX2 vectors before its scalar tail.
    return index < count / 16 * 16 ? ::fmaf(scale, b, a) : optimizer_add(a, optimizer_mul(scale, b));
#endif
}

ELMAN_OPTIMIZER_MATH inline FloatOptimizerElement float_optimizer_element(
        float parameter, float gradient, float first, float second, const FloatOptimizerParameters& p,
        FloatOptimizerAlgorithm algorithm, bool first_step, size_t index, size_t count) {
    gradient = optimizer_mul(gradient, p.scale);
    if (algorithm != FloatOptimizerAlgorithm::AdamW && p.decay != 0.f)
        gradient = optimizer_scaled_add(gradient, parameter, p.decay, index, count);
    if (algorithm == FloatOptimizerAlgorithm::Sgd) {
        const float velocity = p.momentum == 0.f || first_step ? gradient
            : optimizer_add(optimizer_mul(first, p.momentum), gradient);
        return {optimizer_scaled_add(parameter, velocity, -p.rate, index, count), velocity, second};
    }

    const float delta = optimizer_sub(gradient, first);
    const float weight = p.mean_weight;
#ifdef __CUDA_ARCH__
    const float mean = weight < .5f ? ::fmaf(weight, delta, first)
                                    : ::fmaf(-delta, optimizer_sub(1.f, weight), gradient);
    const float square = ::fmaf(p.square_weight, optimizer_mul(gradient, gradient), optimizer_mul(second, p.beta2));
    const float denominator = optimizer_add(optimizer_mul(::sqrtf(square), optimizer_div(1.f, p.correction2_sqrt)), p.epsilon);
#else
    float mean;
    if (index < count / 16 * 16)
        mean = ::fmaf(weight < .5f ? weight : optimizer_sub(weight, 1.f), delta, weight < .5f ? first : gradient);
    else
        mean = weight < .5f ? optimizer_add(first, optimizer_mul(weight, delta))
                            : optimizer_sub(gradient, optimizer_mul(delta, optimizer_sub(1.f, weight)));
    const float square = optimizer_add(optimizer_mul(second, p.beta2),
                                       optimizer_mul(optimizer_mul(p.square_weight, gradient), gradient));
    const float denominator = optimizer_add(optimizer_div(::sqrtf(square), p.correction2_sqrt), p.epsilon);
#endif
    if (algorithm == FloatOptimizerAlgorithm::AdamW) parameter = optimizer_mul(parameter, p.decay_factor);
#ifdef __CUDA_ARCH__
    const float value = ::fmaf(-p.step_size, optimizer_div(mean, denominator), parameter);
#else
    const float value = optimizer_add(parameter, optimizer_div(optimizer_mul(-p.step_size, mean), denominator));
#endif
    return {value, mean, square};
}
} // namespace elman
#undef ELMAN_OPTIMIZER_MATH
