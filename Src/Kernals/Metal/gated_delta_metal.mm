#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_gated_delta(Graph m, const Operands& inputs, const KernelParams& p) {
    const int batch = p.input_shape(KernelInput::A, 0), time = p.input_shape(KernelInput::A, 1);
    const int heads = p.input_shape(KernelInput::A, 2), width = p.attribute(0), columns = p.attribute(1);
    Tensor state = inputs[2];
    NSMutableArray<MPSGraphTensor*>* values = [NSMutableArray arrayWithCapacity:time];
    for (int t = 0; t < time; ++t) {
        Tensor packed = m.reshape(m.slice(inputs[0], 1, t, 1), @[@(batch), @(heads), @(2 * width + columns)]);
        Tensor gates = m.reshape(m.slice(inputs[1], 1, t, 1), @[@(batch), @(heads), @2]);
        Tensor q = m.reshape(m.slice(packed, 2, 0, width), @[@(batch), @(heads), @1, @(width)]);
        Tensor k = m.reshape(m.slice(packed, 2, width, width), @[@(batch), @(heads), @1, @(width)]);
        Tensor v = m.reshape(m.slice(packed, 2, 2 * width, columns), @[@(batch), @(heads), @1, @(columns)]);
        Tensor decay = m.reshape(m.slice(gates, 2, 0, 1), @[@(batch), @(heads), @1, @1]);
        Tensor beta = m.reshape(m.slice(gates, 2, 1, 1), @[@(batch), @(heads), @1, @1]);
        state = m.mul(state, decay);
        Tensor delta = m.mul(m.sub(v, m.matmul(k, state)), beta);
        state = m.add(state, m.matmul(m.transpose(k, 2, 3), delta));
        [values addObject:m.reshape(m.matmul(q, state), @[@(batch), @1, @(heads), @(columns)])];
    }
    Tensor output = [m.g concatTensors:values dimension:1 name:nil];
    return [m.g concatTensors:@[m.reshape(output, @[@(batch * time * heads * columns)]),
                                m.reshape(state, @[@(batch * heads * width * columns)])] dimension:0 name:nil];
}
} // namespace elman::metal
