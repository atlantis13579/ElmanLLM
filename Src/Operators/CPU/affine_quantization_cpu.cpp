#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Operators/affine_quantization_math.h"
#include "Backends/CpuParallel.h"

namespace elman {
namespace {
struct AffineReader {
    double operator()(const uint32_t* values, int index, int dtype) const {
        return dtype == int(DType::Float64) ? reinterpret_cast<const double*>(values)[index] : cpu_read(values, index, dtype);
    }
};
}
void launch_affine_quantization_cpu(AffineQuantization operation, uint32_t* out, const uint32_t* input,
                                    const uint32_t* scales, const uint32_t* zeros, const uint32_t* indices,
                                    const KernelParams& p) {
    const unsigned bits = quantized_storage_bits(DType(p.output_dtype()));
    if (!p.output_count()) return;
    const int rank = p.input_rank(KernelInput::A);
    const int columns = rank ? p.input_shape(KernelInput::A, rank - 1) : 1;
    const int groups = int((int64_t(columns) + p.attribute(0) - 1) / p.attribute(0));
    const int64_t work = operation == AffineQuantization::Dequantize ? int64_t(p.output_count() / columns) * groups :
        (operation == AffineQuantization::Quantize || operation == AffineQuantization::Relu) && bits < 8 ?
        (int64_t(p.output_count()) * bits + 7) / 8 : p.output_count();
    LocalFunction execute{[&](int64_t begin, int64_t end) {
    if (operation == AffineQuantization::Dequantize) {
        const DType code_dtype = DType(p.input_dtype(KernelInput::A));
        const bool wide = p.output_dtype() == int(DType::Float64) || p.input_dtype(KernelInput::B) == int(DType::Float64);
        for (int64_t item = begin; item < end; ++item) {
            const int row = int(item / groups), group = int(item % groups);
            const int scale_index = quantization_parameter_index(KernelInput::B, row, group, p);
            const int zero_index = quantization_parameter_index(KernelInput::C, row, group, p);
            const double scale = AffineReader{}(scales, scale_index, p.input_dtype(KernelInput::B));
            const int64_t zero = affine_zero(zeros, zero_index, p.input_dtype(KernelInput::C));
            const bool valid = std::isfinite(scale) && scale > 0 && zero >= quantized_minimum(code_dtype) && zero <= quantized_maximum(code_dtype);
            const int first = group * p.attribute(0), last = int(std::min(int64_t(columns), int64_t(first) + p.attribute(0)));
            for (int column = first; column < last; ++column) {
                const int index = row * columns + column;
                const int64_t difference = quantized_code(input, uint64_t(index), code_dtype) - (valid ? zero : 0);
                const double value = !valid ? 0 : wide ? double(difference) * scale : double(float(difference) * float(scale));
                if (p.output_dtype() == int(DType::Float64)) reinterpret_cast<double*>(out)[index] = value;
                else cpu_write(out, index, float(value), p.output_dtype());
            }
        }
        return;
    }
    for (int64_t i = begin; i < end; ++i) {
        if (operation == AffineQuantization::Quantize || operation == AffineQuantization::Relu) affine_quantize_unit(int(i), operation, out, input, scales, zeros, p, AffineReader{});
        else {
            const double value = affine_decode_element(int(i), operation, input, scales, zeros, indices, p, AffineReader{});
            if (p.output_dtype() == int(DType::Float64)) reinterpret_cast<double*>(out)[i] = value;
            else cpu_write(out, int(i), float(value), p.output_dtype());
        }
    }
    }};
    parallel_for_cpu(work, 262144, execute, 32);
}
void launch_affine_quantize_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    launch_affine_quantization_cpu(AffineQuantization::Quantize, out, a, b, c, nullptr, p);
}
void launch_affine_relu_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    launch_affine_quantization_cpu(AffineQuantization::Relu, out, a, b, c, nullptr, p);
}
void launch_affine_dequantize_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    launch_affine_quantization_cpu(AffineQuantization::Dequantize, out, a, b, c, nullptr, p);
}
void launch_affine_embedding_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* d, const KernelParams& p) {
    launch_affine_quantization_cpu(AffineQuantization::Embedding, out, a, b, c, d, p);
}
} // namespace elman
