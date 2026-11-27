#include "kernel.h"
#include "Kernals/packed_linear_params.h"

namespace elman {
void launch_linear_cpu(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
void launch_packed_linear_cpu(uint32_t* out, const uint32_t* x, const uint32_t* weight,
                              const uint32_t* bias, const KernelParams& p) {
    if (!p.output_count()) return;
    size_t offset = 0, width_bytes = dtype_storage_bytes(DType(p.output_dtype()), 1);
    int inputs = p.input_shape(KernelInput::B, 1), columns = p.input_shape(KernelInput::B, 0);
    int rows = p.output_count() / columns;
    for (int group = 0; group < p.attribute(kernal_attr::packed_linear::kCount); ++group) {
        int width = p.attribute(kernal_attr::packed_linear::kWidths + group);
        if (width) launch_linear_cpu(projection_offset(out, offset * rows, width_bytes), x,
            projection_offset(weight, offset * inputs, width_bytes),
            p.input_rank(KernelInput::C) < 0 ? nullptr : projection_offset(bias, offset, width_bytes),
            projection_params(p, width));
        offset += width;
    }
}
} // namespace elman
