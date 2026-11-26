#pragma once

#include "Kernals/complex_pointwise.h"
#include <cmath>
#ifdef __CUDACC__
#include <thrust/complex.h>
#define ELMAN_COMPLEX_FN __device__
#else
#include <complex>
#define ELMAN_COMPLEX_FN
#endif

namespace elman {
#ifdef __CUDACC__
namespace complex_library = thrust;
#else
namespace complex_library = std;
#endif
template<class Real> using ComplexNumber = complex_library::complex<Real>;

template<class Real>
ELMAN_COMPLEX_FN ComplexNumber<Real> complex_product(ComplexNumber<Real> a, ComplexNumber<Real> b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}
template<class Real>
ELMAN_COMPLEX_FN ComplexNumber<Real> complex_quotient(ComplexNumber<Real> numerator, ComplexNumber<Real> denominator) {
    const Real real = denominator.real(), imag = denominator.imag();
    const Real real_size = ::fabs(real), imag_size = ::fabs(imag);
    if (real_size == 0 && imag_size == 0) return {numerator.real() / real_size, numerator.imag() / imag_size};
    // Normalize by the larger component to avoid squaring the denominator.
    if (real_size >= imag_size) {
        const Real ratio = imag / real;
        const Real inverse = Real(1) / (real + imag * ratio);
        return {(numerator.real() + numerator.imag() * ratio) * inverse,
                (numerator.imag() - numerator.real() * ratio) * inverse};
    }
    const Real ratio = real / imag;
    const Real inverse = Real(1) / (imag + real * ratio);
    return {(numerator.real() * ratio + numerator.imag()) * inverse,
            (numerator.imag() * ratio - numerator.real()) * inverse};
}
template<class Real>
ELMAN_COMPLEX_FN ComplexNumber<Real> complex_pointwise_value(ComplexPointwise operation, ComplexNumber<Real> a,
                                                             ComplexNumber<Real> b, ComplexNumber<Real> c) {
    using Number = ComplexNumber<Real>;
    const bool a_true = a.real() != 0 || a.imag() != 0;
    const bool b_true = b.real() != 0 || b.imag() != 0;
    switch (operation) {
    case ComplexPointwise::Cast: return a;
    case ComplexPointwise::Add: return {a.real() + b.real(), a.imag() + b.imag()};
    case ComplexPointwise::Sub: return {a.real() - b.real(), a.imag() - b.imag()};
    case ComplexPointwise::Mul: return complex_product(a, b);
    case ComplexPointwise::Div: return complex_quotient(a, b);
    case ComplexPointwise::Pow: return complex_library::pow(a, b);
    case ComplexPointwise::Conj: return {a.real(), -a.imag()};
    case ComplexPointwise::Real: return {a.real(), 0};
    case ComplexPointwise::Imag: return {a.imag(), 0};
    case ComplexPointwise::Abs: return {complex_library::abs(a), 0};
    case ComplexPointwise::Angle: return {complex_library::arg(a), 0};
    case ComplexPointwise::Sgn: return a_true ? a / complex_library::abs(a) : Number(0, 0);
    case ComplexPointwise::Sqrt: return complex_library::sqrt(a);
    case ComplexPointwise::Rsqrt: return complex_quotient(Number(1, 0), complex_library::sqrt(a));
    case ComplexPointwise::Exp: return complex_library::exp(a);
    case ComplexPointwise::Log: return complex_library::log(a);
    case ComplexPointwise::Sin: return complex_library::sin(a);
    case ComplexPointwise::Cos: return complex_library::cos(a);
    case ComplexPointwise::Tanh: return complex_library::tanh(a);
    case ComplexPointwise::Eq: return {Real(a.real() == b.real() && a.imag() == b.imag()), 0};
    case ComplexPointwise::Ne: return {Real(a.real() != b.real() || a.imag() != b.imag()), 0};
    case ComplexPointwise::LogicalAnd: return {Real(a_true && b_true), 0};
    case ComplexPointwise::LogicalOr: return {Real(a_true || b_true), 0};
    case ComplexPointwise::LogicalNot: return {Real(!a_true), 0};
    case ComplexPointwise::Where: return a_true ? b : c;
    case ComplexPointwise::Construct: return {a.real(), b.real()};
    default: return {};
    }
}
} // namespace elman
#undef ELMAN_COMPLEX_FN
