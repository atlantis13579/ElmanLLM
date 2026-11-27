#include "kernel.h"
#include "Kernals/index_copy_params.h"

namespace elman {
void launch_index_copy_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                           const uint32_t* c, const KernelParams& p) {
    if (!p.output_count()) return;
    if (out != a) std::memcpy(out, a, dtype_storage_bytes(DType(p.output_dtype()), size_t(p.output_count())));
    int axis = p.attribute(kernal_attr::index_copy::kAxis);
    int inner = p.input_stride(KernelInput::C, axis), length = p.input_shape(KernelInput::C, axis);
    int destination_length = p.input_shape(KernelInput::A, axis);
    for (int i = 0, count = input_count(KernelInput::C, p); i < count; ++i) {
        int64_t selected = index_copy_index(b, i / inner % length, p.input_dtype(KernelInput::B));
        if (selected >= 0 && selected < destination_length) {
            int target = i / (inner * length) * destination_length * inner + int(selected) * inner + i % inner;
            COPY_TYPED_ELEMENT(out, c, target, i, p.output_dtype());
        }
    }
}
} // namespace elman
