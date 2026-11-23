#include "Optimizer/OptimizerInternal.h"
#include "Core/LocalFunction.h"
#include <cmath>

namespace elman::optimizer_internal {
TensorStep complex_step(const std::string& algorithm, const Tensor& values, const Tensor& gradients,
                        const Tensor& first_moment, const Tensor& second_moment,
                        const StepParameters& step, bool first_step) {
    LocalFunction component{[&](const Tensor& value, const Tensor& input_gradient,
                         const Tensor& first, const Tensor& second) {
        Tensor gradient = mul(input_gradient, step.gradient_scale);
        if (algorithm != "adamw" && step.weight_decay != 0.) gradient = add(gradient, mul(value, step.weight_decay));
        TensorStep result;
        if (algorithm == "sgd") {
            result.first_moment = step.momentum == 0. || first_step ? gradient : add(mul(first, step.momentum), gradient);
            result.second_moment = second;
            result.value = sub(value, mul(result.first_moment, step.rate));
            return result;
        }
        const Tensor delta = sub(gradient, first);
        result.first_moment = 1. - step.beta1 < .5 ? add(first, mul(delta, 1. - step.beta1)) : sub(gradient, mul(delta, step.beta1));
        result.second_moment = add(mul(second, step.beta2), mul(mul(gradient, gradient), 1. - step.beta2));
        const Tensor denominator = add(div(sqrt(result.second_moment), std::sqrt(step.correction2)), step.epsilon);
        const Tensor parameter = algorithm == "adamw" && step.weight_decay != 0. ? mul(value, 1. - step.rate * step.weight_decay) : value;
        result.value = sub(parameter, div(mul(result.first_moment, step.rate / step.correction1), denominator));
        return result;
    }};
    // Apply the optimizer independently to the real and imaginary components.
    const TensorStep r = component(real(values), real(gradients), real(first_moment), real(second_moment));
    const TensorStep i = component(imag(values), imag(gradients), imag(first_moment), imag(second_moment));
    return {complex(r.value, i.value), complex(r.first_moment, i.first_moment), complex(r.second_moment, i.second_moment)};
}
} // namespace elman::optimizer_internal
