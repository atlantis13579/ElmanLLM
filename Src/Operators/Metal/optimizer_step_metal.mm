#include "Backends/MPSGraph.h"

#include "Operators/low_precision_optimizer.h"

namespace elman::metal {
Tensor lower_optimizer_step(Graph m, const Operands& inputs, const KernelParams& p) {
  auto w = inputs[0], grad = inputs[1], first = inputs[2], second = inputs[3];
  using namespace optimizer_attr;
  const int algorithm = p.attribute(kAlgorithm), mode = p.attribute(kResult);
  auto parameter = [&](int i) { return optimizer_parameter(p, i); };
  const double rate = parameter(0), decay = parameter(2);
  auto update = m.mul(grad, m.number(parameter(1)));
  if (algorithm != 2 && decay != 0) {
    update = m.add(update, m.mul(w, m.number(decay)));
  }
  if (algorithm == 0) {
    auto momentum = parameter(3);
    auto velocity =
        momentum == 0 || p.attribute(kFirstStep) ? update : m.add(m.mul(first, m.number(momentum)), update);
    if (mode == 1) {
      return velocity;
    }
    if (mode == 2) {
      return second;
    }
    return m.sub(w, m.mul(velocity, m.number(rate)));
  }
  auto beta1 = parameter(4), beta2 = parameter(5);
  auto moving_mean = m.add(m.mul(first, m.number(beta1)), m.mul(update, m.number(1 - beta1)));
  auto moving_square = m.add(m.mul(second, m.number(beta2)), m.mul(m.square(update), m.number(1 - beta2)));
  if (mode == 1) {
    return moving_mean;
  }
  if (mode == 2) {
    return moving_square;
  }
  auto corrected = m.add(m.div(m.sqrt(moving_square), m.number(parameter(8))), m.number(parameter(6)));
  auto weights = algorithm == 2 && decay != 0 ? m.mul(w, m.number(1. - rate * decay)) : w;
  return m.sub(weights, m.mul(m.div(moving_mean, corrected), m.number(rate / parameter(7))));
}
}  // namespace elman::metal
