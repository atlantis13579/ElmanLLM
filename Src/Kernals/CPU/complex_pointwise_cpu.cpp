#include "kernel.h"
#include "Kernals/complex_storage.h"

namespace elman {
namespace {
struct Reader {
    double operator()(const uint32_t* data, uint64_t index, DType dtype) const {
        if (dtype == DType::Float64) return reinterpret_cast<const double*>(data)[index];
        if (dtype == DType::Float32) return reinterpret_cast<const float*>(data)[index];
        const uint16_t bits = reinterpret_cast<const uint16_t*>(data)[index];
        return dtype == DType::Float16 ? numeric::half_to_float(bits) : numeric::bfloat_to_float(bits);
    }
};
struct Writer {
    void operator()(uint32_t* data, uint64_t index, DType dtype, double value) const {
        if (dtype == DType::Float64) reinterpret_cast<double*>(data)[index] = value;
        else if (dtype == DType::Float32) reinterpret_cast<float*>(data)[index] = float(value);
        else reinterpret_cast<uint16_t*>(data)[index] = dtype == DType::Float16 ?
            numeric::float_to_half(float(value)) : numeric::float_to_bfloat(float(value));
    }
};
template<class Real>
void execute(ComplexPointwise operation, uint32_t* out, const uint32_t* a, const uint32_t* b,
              const uint32_t* c, const KernelParams& p) {
    const int arity = complex_pointwise_arity(operation);
    for (int i = 0; i < p.output_count(); ++i) {
        const ComplexNumber<Real> x = complex_read<Real>(a, uint64_t(broadcast_index(i, KernelInput::A, p)), DType(p.input_dtype(KernelInput::A)), Reader{});
        const ComplexNumber<Real> y = arity > 1 ? complex_read<Real>(b, uint64_t(broadcast_index(i, KernelInput::B, p)), DType(p.input_dtype(KernelInput::B)), Reader{}) : ComplexNumber<Real>{};
        const ComplexNumber<Real> z = arity > 2 ? complex_read<Real>(c, uint64_t(broadcast_index(i, KernelInput::C, p)), DType(p.input_dtype(KernelInput::C)), Reader{}) : ComplexNumber<Real>{};
        complex_write(out, uint64_t(i), DType(p.output_dtype()), complex_pointwise_value(operation, x, y, z), Writer{});
    }
}
}
void launch_complex_pointwise_cpu(ComplexPointwise operation, uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (complex_pointwise_wide(p)) execute<double>(operation, out, a, b, c, p);
    else execute<float>(operation, out, a, b, c, p);
}
} // namespace elman
