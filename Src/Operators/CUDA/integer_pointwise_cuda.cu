#include "kernel.cuh"
#include "Operators/integer_pointwise.h"

namespace elman {
namespace {
__device__ double read_float(const uint32_t* data, int index, int dtype) {
    return dtype == int(DType::Float64) ? reinterpret_cast<const double*>(data)[index] :
        elman_cuda_decode(elman_cuda_load(data, index, dtype), dtype);
}
__device__ uint64_t float_to_integer(double value, int dtype, int source_dtype) {
    if (dtype == int(DType::Bool)) return value != 0;
    if (dtype == int(DType::Int64) || dtype == int(DType::UInt8)) return uint64_t(__double2ll_rz(value));
    if (dtype == int(DType::UInt64)) return __double2ull_rz(value);
    if (dtype == int(DType::UInt16) || dtype == int(DType::UInt32))
        return source_dtype == int(DType::Float64) ? __double2uint_rz(value) : __float2uint_rz(float(value));
    // CUDA's float and double conversions have different NaN results.
    return uint64_t(int64_t(source_dtype == int(DType::Float64) ? __double2int_rz(value) : __float2int_rz(float(value))));
}
__device__ uint64_t read_value(const uint32_t* data, int index, int dtype) {
    return is_standard_integer(DType(dtype)) ? integer_reduction_value(data, index, dtype) : uint64_t(read_float(data, index, dtype) != 0);
}
__global__ void kernel_integer_pointwise(IntegerPointwise op, uint32_t* out, const uint32_t* a,
                                         const uint32_t* b, const uint32_t* c, KernelParams p) {
    const unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= unsigned(p.output_count())) return;
    const int i = int(element), ai = broadcast_index(i, KernelInput::A, p);
    const int ad = p.input_dtype(KernelInput::A), od = p.output_dtype();
    if (op == IntegerPointwise::Cast) {
        if (!is_standard_integer(DType(ad))) integer_write(out, i, od, float_to_integer(read_float(a, ai, ad), od, ad));
        else {
            const uint64_t value = integer_reduction_value(a, ai, ad);
            if (is_standard_integer(DType(od))) integer_write(out, i, od, value);
            else if (od == int(DType::Float64)) reinterpret_cast<double*>(out)[i] = integer_to_float<double>(value, ad);
            else elman_cuda_store(out, i, elman_cuda_encode(integer_to_float<float>(value, ad), od), od);
        }
        return;
    }
    const uint64_t x = read_value(a, ai, ad);
    const uint64_t y = integer_pointwise_arity(op) >= 2 ? read_value(b, broadcast_index(i, KernelInput::B, p), p.input_dtype(KernelInput::B)) : 0;
    const uint64_t z = integer_pointwise_arity(op) >= 3 ? read_value(c, broadcast_index(i, KernelInput::C, p), p.input_dtype(KernelInput::C)) : 0;
    integer_write(out, i, od, integer_pointwise_value(op, x, y, z, ad));
}
}
void launch_integer_pointwise_cuda(IntegerPointwise op, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (p.output_count() > 0)
        kernel_integer_pointwise<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(op, out, a, b, c, p);
}
} // namespace elman
