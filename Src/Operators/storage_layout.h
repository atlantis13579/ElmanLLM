#pragma once
#include "Core/Backend.h"
#include "Operators/index_copy_params.h"

#ifdef __CUDACC__
#define ELMAN_LAYOUT_HD __host__ __device__
#else
#define ELMAN_LAYOUT_HD
#endif
namespace elman {
enum class StorageLayout { None, Copy, Permute, Transpose, Slice, Repeat, Expand, Concat, Gather, IndexSelect, IndexCopy, Triu, Tril, SliceGradient };

inline StorageLayout storage_layout_operation(const std::string& op, DType dtype) {
    // Existing numeric kernels retain their established mixed-dtype raw ABI.
    if (is_compute_float(dtype) || kernel_integer(int(dtype))) return StorageLayout::None;
    if (op == "copy" || op == "reshape" || op == "squeeze" || op == "unsqueeze") return StorageLayout::Copy;
    if (op == "permute") return StorageLayout::Permute;
    if (op == "transpose") return StorageLayout::Transpose;
    if (op == "slice" || op == "split") return StorageLayout::Slice;
    if (op == "repeat") return StorageLayout::Repeat;
    if (op == "expand") return StorageLayout::Expand;
    if (op == "concat") return StorageLayout::Concat;
    if (op == "gather") return StorageLayout::Gather;
    if (op == "index_select") return StorageLayout::IndexSelect;
    if (op == "index_copy") return StorageLayout::IndexCopy;
    if ((op == "slice_grad" || op == "slice_backward") && (is_complex(dtype) || dtype == DType::Float64 || is_low_precision_float(dtype))) return StorageLayout::SliceGradient;
    // E8M0 masks encode zero as the smallest representable value.
    // Quantized formats still require their scale and zero-point metadata.
    const bool zero = dtype_info(dtype).kind != DTypeKind::Quantized;
    if (op == "triu" && zero) return StorageLayout::Triu;
    if (op == "tril" && zero) return StorageLayout::Tril;
    return StorageLayout::None;
}

struct StorageLocation { int input, index; };
ELMAN_LAYOUT_HD inline StorageLocation storage_layout_source(StorageLayout op, int index,
                                                              const uint32_t* indices, const KernelParams& p) {
    if (op == StorageLayout::Copy) return {0, index};
    if (op == StorageLayout::Triu || op == StorageLayout::Tril) {
        const int columns = p.output_shape(p.output_rank() - 1), rows = p.output_shape(p.output_rank() - 2);
        const int row = index / columns % rows, column = index % columns;
        const bool keep = op == StorageLayout::Triu ? column - row >= p.attribute(0) : column - row <= p.attribute(0);
        return {keep ? 0 : -1, index};
    }
    const int axis = p.attribute(0);
    if (op == StorageLayout::Concat) {
        const int inner = p.input_stride(KernelInput::A, axis), full = p.output_shape(axis);
        const int coordinate = index / inner % full, outer = index / (inner * full), rest = index % inner;
        const int left = p.input_shape(KernelInput::A, axis);
        const int input = coordinate < left ? 0 : 1;
        const int length = p.input_shape(KernelInput(input), axis);
        return {input, outer * length * inner + (coordinate - (input ? left : 0)) * inner + rest};
    }
    if (op == StorageLayout::IndexCopy) {
        const int inner = p.input_stride(KernelInput::A, axis), length = p.input_shape(KernelInput::A, axis);
        const int coordinate = index / inner % length, source_length = p.input_shape(KernelInput::C, axis);
        // Last duplicate wins, matching the existing deterministic indexed-copy contract.
        for (int j = source_length - 1; j >= 0; --j)
            if (index_copy_index(indices, j, p.input_dtype(KernelInput::B)) == coordinate)
                return {2, index / (inner * length) * source_length * inner + j * inner + index % inner};
        return {0, index};
    }
    int flat = index, source = 0;
    const int delta = p.output_rank() - p.input_rank(KernelInput::A);
    for (int d = p.output_rank() - 1; d >= 0; --d) {
        int coordinate = flat % p.output_shape(d);
        flat /= p.output_shape(d);
        int source_axis = d;
        if (op == StorageLayout::Permute) source_axis = p.attribute(d);
        else if (op == StorageLayout::Transpose)
            source_axis = d == p.attribute(0) ? p.attribute(1) : d == p.attribute(1) ? p.attribute(0) : d;
        else if (op == StorageLayout::Repeat || op == StorageLayout::Expand) {
            source_axis -= delta;
            if (source_axis < 0) continue;
            coordinate = op == StorageLayout::Repeat ? coordinate % p.input_shape(KernelInput::A, source_axis) :
                p.input_shape(KernelInput::A, source_axis) == 1 ? 0 : coordinate;
        } else if (d == axis) {
            if (op == StorageLayout::Slice) coordinate = p.attribute(1) + coordinate * p.attribute(2);
            else if (op == StorageLayout::SliceGradient) {
                coordinate -= p.attribute(1);
                if (coordinate < 0 || coordinate % p.attribute(2)) return {-1, 0};
                coordinate /= p.attribute(2);
                if (coordinate >= p.input_shape(KernelInput::A, d)) return {-1, 0};
            }
            else if (op == StorageLayout::Gather) coordinate = reinterpret_cast<const int32_t*>(indices)[index];
            else if (op == StorageLayout::IndexSelect) coordinate = reinterpret_cast<const int32_t*>(indices)[coordinate];
        }
        source += coordinate * p.input_stride(KernelInput::A, source_axis);
    }
    return {0, source};
}
ELMAN_LAYOUT_HD inline uint8_t storage_layout_byte(StorageLayout op, uint64_t byte, unsigned bits,
                                                   const uint32_t* a, const uint32_t* b, const uint32_t* c,
                                                   const KernelParams& p) {
    const uint8_t* inputs[] = {reinterpret_cast<const uint8_t*>(a), reinterpret_cast<const uint8_t*>(b), reinterpret_cast<const uint8_t*>(c)};
    if (op == StorageLayout::Copy) return inputs[0][byte];
    if (bits % 8 == 0) {
        const unsigned width = bits / 8;
        const auto source = storage_layout_source(op, int(byte / width), b, p);
        return source.input < 0 ? 0 : inputs[source.input][uint64_t(source.index) * width + byte % width];
    }
    // Each thread owns one destination byte, including cross-byte packed values.
    // This avoids read/modify/write races and never accesses padding allocation.
    uint8_t result = 0;
    for (unsigned bit = 0; bit < 8; ++bit) {
        const uint64_t logical_bit = byte * 8 + bit, element = logical_bit / bits;
        if (element >= uint64_t(p.output_count())) break;
        const auto source = storage_layout_source(op, int(element), b, p);
        if (source.input < 0) continue;
        const uint64_t source_bit = uint64_t(source.index) * bits + logical_bit % bits;
        result |= ((inputs[source.input][source_bit / 8] >> (source_bit % 8)) & 1u) << bit;
    }
    return result;
}
void validate_storage_layout(StorageLayout op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_storage_layout_cpu(StorageLayout op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_storage_layout_cuda(StorageLayout op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
#undef ELMAN_LAYOUT_HD
