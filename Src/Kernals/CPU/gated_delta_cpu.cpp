#include "kernel.h"
#include "Backends/CpuParallel.h"

namespace elman {
void launch_gated_delta_cpu(uint32_t* output, const uint32_t* packed, const uint32_t* gates,
                            const uint32_t* initial, const KernelParams& p) {
    const float* a = reinterpret_cast<const float*>(packed);
    const float* b = reinterpret_cast<const float*>(gates);
    const float* c = reinterpret_cast<const float*>(initial);
    float* out = reinterpret_cast<float*>(output);
    const int batch = p.input_shape(KernelInput::A, 0), time = p.input_shape(KernelInput::A, 1);
    const int heads = p.input_shape(KernelInput::A, 2), width = p.attribute(0), columns = p.attribute(1);
    const int output_count = batch * time * heads * columns, packed_width = 2 * width + columns;
    parallel_for_cpu(int64_t(batch) * heads, 1, [&](int64_t begin, int64_t end) {
        for (int group = int(begin); group < end; ++group) {
            float* state = out + output_count + group * width * columns;
            std::copy_n(c + group * width * columns, width * columns, state);
            for (int t = 0; t < time; ++t) {
                const int row = (group / heads * time + t) * heads + group % heads;
                const float* q = a + row * packed_width; const float* k = q + width; const float* v = k + width;
                const float decay = b[row * 2], beta = b[row * 2 + 1];
                for (int column = 0; column < columns; ++column) {
                    float prediction = 0;
                    for (int d = 0; d < width; ++d) {
                        state[d * columns + column] *= decay;
                        prediction += state[d * columns + column] * k[d];
                    }
                    const float delta = (v[column] - prediction) * beta;
                    float value = 0;
                    for (int d = 0; d < width; ++d) {
                        state[d * columns + column] += k[d] * delta;
                        value += state[d * columns + column] * q[d];
                    }
                    out[row * columns + column] = value;
                }
            }
        }
    });
}
} // namespace elman
