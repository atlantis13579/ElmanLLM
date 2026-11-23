#pragma once

#include "Core/Backend.h"
#include <stdexcept>

namespace elman::metal {
inline int wide_words(DType type) {
    return type == DType::Float64 ? 2 : type == DType::Complex128 ? 4 : 0;
}

inline bool raw_layout(const std::string& op) {
    return op == "copy" || op == "reshape" || op == "squeeze" || op == "unsqueeze" ||
        op == "expand" || op == "transpose" || op == "permute" || op == "concat" || op == "repeat" ||
        op == "slice" || op == "split" || op == "tril" || op == "triu" || op == "where" ||
        op == "gather" || op == "index_select" || op == "scatter" || op == "index_copy" || op == "sort_backward";
}

inline void validate_operator_dtypes(const std::string& op, const KernelParams& p) {
    const DType out = DType(p.output_dtype());
    DType widest = out;
    for (KernelInput slot : {KernelInput::A, KernelInput::B, KernelInput::C, KernelInput::D}) {
        const DType type = DType(p.input_dtype(slot));
        if (p.input_rank(slot) >= 0 && wide_words(type) > wide_words(widest)) widest = type;
    }
    if (!wide_words(widest)) return;

    const DType a = DType(p.input_dtype(KernelInput::A));
    const DType b = DType(p.input_dtype(KernelInput::B));
    const DType c = DType(p.input_dtype(KernelInput::C));
    if ((op == "real" || op == "imag") && a == DType::Complex128 && out == DType::Float64) return;
    if (op == "complex" && a == DType::Float64 && b == a && out == DType::Complex128) return;
    if (op == "conj" && out == DType::Complex128 && a == out) return;
    if (op == "where" && a == DType::Bool && wide_words(out) && b == out && c == out) return;
    if (wide_words(out) && a == out && op != "where" &&
        (raw_layout(op) || op == "slice_grad" || op == "slice_backward" || op == "cast")) return;
    throw std::invalid_argument(std::string("Metal does not support ") + dtype_name(widest) + " arithmetic for " + op);
}
} // namespace elman::metal
