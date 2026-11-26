#include "kernel.cuh"
#include "Core/AllocationPool.h"
#include "Kernals/index_copy_params.h"
#include <stdexcept>
#include <string>

namespace elman {
void launch_argsort_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
namespace {
__global__ void narrow_indices(uint32_t* out, const uint32_t* indices, int count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < unsigned(count)) out[i] = uint32_t(reinterpret_cast<const int64_t*>(indices)[i]);
}
__global__ void copy_indexed(uint32_t* out, const uint32_t* indices, const uint32_t* source,
                             const uint32_t* order, int count, KernelParams p) {
    unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= unsigned(count)) return;
    int i = int(element), axis = p.attribute(operator_attr::index_copy::kAxis);
    int inner = p.input_stride(KernelInput::C, axis), length = p.input_shape(KernelInput::C, axis);
    int position = i / inner % length;
    int selected_source = order ? int(order[position]) : position;
    int dtype = p.input_dtype(KernelInput::B), destination_length = p.input_shape(KernelInput::A, axis);
    int64_t selected = index_copy_index(indices, selected_source, dtype);
    if (selected < 0 || selected >= destination_length) return;
    // Stable index sorting gives each duplicated destination one final writer.
    if (order && position + 1 < length && index_copy_index(indices, int(order[position + 1]), dtype) == selected) return;
    int outer = i / (inner * length), column = i % inner;
    int target = outer * destination_length * inner + int(selected) * inner + column;
    int input = outer * length * inner + selected_source * inner + column;
    int output_dtype = p.output_dtype();
    uint32_t word = elman_cuda_load(source, input, output_dtype);
    // CUDA indexed assignment canonicalizes narrow
    // NaNs through opmath conversion. Unselected base elements remain intact.
    if ((output_dtype == int(DType::Float16) && (word & 0x7fffu) > 0x7c00u) ||
        (output_dtype == int(DType::BFloat16) && (word & 0x7fffu) > 0x7f80u)) word = 0x7fffu;
    elman_cuda_store(out, target, word, output_dtype);
}
}
void launch_index_copy_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
    if (!p.output_count()) return;
    if (out != a) {
        cudaError_t result = cudaMemcpyAsync(out, a, dtype_storage_bytes(DType(p.output_dtype()), size_t(p.output_count())),
                                      cudaMemcpyDeviceToDevice, current_cuda_stream());
        if (result != cudaSuccess) throw std::runtime_error(std::string("CUDA index copy: ") + cudaGetErrorString(result));
    }
    int count = p.input_shape(KernelInput::C, 0) * p.input_stride(KernelInput::C, 0);
    if (!count) return;
    BufferPtr order, narrowed;
    const uint32_t* keys = b;
    if (p.attribute(operator_attr::index_copy::kHasDuplicates)) {
        int length = p.input_shape(KernelInput::B, 0);
        order = backend_workspace(Backend::CUDA, size_t(length) * sizeof(int32_t));
        if (p.input_dtype(KernelInput::B) == int(DType::Int64)) {
            narrowed = backend_workspace(Backend::CUDA, size_t(length) * sizeof(int32_t));
            uint32_t* values = static_cast<uint32_t*>(narrowed->raw_pointer());
            narrow_indices<<<(unsigned(length) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(values, b, length);
            keys = values;
        }
        KernelParams sort;
        sort.v[KernelParams::kOutputCountSlot] = length;
        sort.v[KernelParams::kOutputRankSlot] = 1;
        sort.v[KernelParams::kOutputShapeSlot] = length;
        sort.v[KernelParams::kOutputDTypeSlot] = int(DType::Int32);
        for (KernelInput input : {KernelInput::A, KernelInput::B, KernelInput::C, KernelInput::D})
            sort.v[KernelParams::input_slot(input)] = -1;
        int slot = KernelParams::input_slot(KernelInput::A);
        sort.v[slot] = 1;
        sort.v[slot + KernelParams::kInputShapeOffset] = length;
        sort.v[slot + KernelParams::kInputStrideOffset] = 1;
        sort.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Int32);
        launch_argsort_cuda(static_cast<uint32_t*>(order->raw_pointer()), keys, nullptr, nullptr, sort);
    }
    copy_indexed<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, b, c,
        order ? static_cast<const uint32_t*>(order->raw_pointer()) : nullptr, count, p);
}
} // namespace elman
