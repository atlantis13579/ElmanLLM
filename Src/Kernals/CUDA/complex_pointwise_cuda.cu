#include "kernel.cuh"
#include "Kernals/complex_storage.h"

namespace elman {
namespace {
struct Reader {
    __device__ double operator()(const uint32_t* data, uint64_t index, DType dtype) const {
        if (dtype == DType::Float64) return reinterpret_cast<const double*>(data)[index];
        if (dtype == DType::Float32) return reinterpret_cast<const float*>(data)[index];
        return elman_cuda_decode(reinterpret_cast<const uint16_t*>(data)[index], int(dtype));
    }
};
struct Writer {
    __device__ void operator()(uint32_t* data, uint64_t index, DType dtype, double value) const {
        if (dtype == DType::Float64) reinterpret_cast<double*>(data)[index] = value;
        else if (dtype == DType::Float32) reinterpret_cast<float*>(data)[index] = float(value);
        else reinterpret_cast<uint16_t*>(data)[index] = uint16_t(elman_cuda_encode(float(value), int(dtype)));
    }
};
template<class Real>
__global__ void complex_kernel(ComplexPointwise operation, uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c, KernelParams p, int arity) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= uint64_t(p.output_count())) return;
    const int i = int(index);
    const ComplexNumber<Real> x = complex_read<Real>(a, uint64_t(broadcast_index(i, KernelInput::A, p)), DType(p.input_dtype(KernelInput::A)), Reader{});
    const ComplexNumber<Real> y = arity > 1 ? complex_read<Real>(b, uint64_t(broadcast_index(i, KernelInput::B, p)), DType(p.input_dtype(KernelInput::B)), Reader{}) : ComplexNumber<Real>{};
    const ComplexNumber<Real> z = arity > 2 ? complex_read<Real>(c, uint64_t(broadcast_index(i, KernelInput::C, p)), DType(p.input_dtype(KernelInput::C)), Reader{}) : ComplexNumber<Real>{};
    complex_write(out, index, DType(p.output_dtype()), complex_pointwise_value(operation, x, y, z), Writer{});
}
}
void launch_complex_pointwise_cuda(ComplexPointwise operation, uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (!p.output_count()) return;
    const unsigned blocks = (unsigned(p.output_count()) + 255) / 256;
    const int arity = complex_pointwise_arity(operation);
    if (complex_pointwise_wide(p)) complex_kernel<double><<<blocks, 256, 0, current_cuda_stream()>>>(operation, out, a, b, c, p, arity);
    else complex_kernel<float><<<blocks, 256, 0, current_cuda_stream()>>>(operation, out, a, b, c, p, arity);
}
} // namespace elman
