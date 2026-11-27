#include "Backends/MPSGraph.h"
#include "Core/LocalFunction.h"

#include "Backends/MetalDTypeSupport.h"
#include "Core/LowPrecisionFloat.h"

#include <cstring>
#include <unordered_map>

namespace elman::metal {
using Lowering = Tensor (*)(Graph, const Operands&, const KernelParams&);

#define ELMAN_METAL_KERNEL(name) Tensor lower_##name(Graph, const Operands&, const KernelParams&);
#include "Kernals/metal_dispatch.inc"
#undef ELMAN_METAL_KERNEL

namespace {
unsigned packed_width(DType type) {
  if (type == DType::Float4E2M1FN || type == DType::QUInt4) {
    return 4;
  }
  if (type == DType::QUInt2) {
    return 2;
  }
  return packed_integer_bits(type);
}
}  // namespace
MPSShape* shape(const KernelParams& p, int input) {
  const KernelInput operand = static_cast<KernelInput>(input);
  const int rank = input < 0 ? p.output_rank() : p.input_rank(operand);
  if (rank == 0) {
    return @[ @1 ];
  }
  NSMutableArray<NSNumber*>* dims = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  for (int axis = 0; axis < rank; ++axis) {
    [dims addObject:@(input < 0 ? p.output_shape(axis) : p.input_shape(operand, axis))];
  }
  return dims;
}

MPSShape* storage_shape(const KernelParams& p, int input) {
  const DType type = DType(input < 0 ? p.output_dtype() : p.input_dtype(static_cast<KernelInput>(input)));
  const unsigned bits = packed_width(type);
  Shape dims = shape(p, input);
  const int words = wide_words(type);
  if (!bits && !words) {
    return dims;
  }
  size_t count = 1;
  for (NSNumber* dim in dims) {
    count *= dim.unsignedIntegerValue;
  }
  if (words) {
    return @[ @(count), @(words) ];
  }
  return @[ @(dtype_storage_bytes(type, count)) ];
}

MPSDataType storage_type(DType type) {
  if (wide_words(type)) {
    return MPSDataTypeInt32;
  }
  if (is_float8(type) || packed_width(type)) {
    return MPSDataTypeUInt8;
  }
  switch (type) {
    case DType::Float32:
      return MPSDataTypeFloat32;
    case DType::Float16:
      return MPSDataTypeFloat16;
    case DType::BFloat16:
      return MPSDataTypeBFloat16;
    case DType::Bool:
      return MPSDataTypeBool;
    case DType::Int8:
    case DType::QInt8:
      return MPSDataTypeInt8;
    case DType::UInt8:
    case DType::QUInt8:
      return MPSDataTypeUInt8;
    case DType::Int16:
      return MPSDataTypeInt16;
    case DType::UInt16:
      return MPSDataTypeUInt16;
    case DType::Int32:
    case DType::QInt32:
      return MPSDataTypeInt32;
    case DType::UInt32:
      return MPSDataTypeUInt32;
    case DType::Int64:
      return MPSDataTypeInt64;
    case DType::UInt64:
      return MPSDataTypeUInt64;
    case DType::Complex32:
      return MPSDataTypeComplexFloat16;
    case DType::Complex64:
      return MPSDataTypeComplexFloat32;
    default:
      return MPSDataTypeInvalid;
  }
}

float attribute_float(const KernelParams& p, int slot) {
  int bits = p.attribute(slot);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
NSArray<NSNumber*>* axes(int start, int end) {
  NSMutableArray<NSNumber*>* result = [NSMutableArray<NSNumber*> arrayWithCapacity:end - start];
  for (int axis = start; axis < end; ++axis) {
    [result addObject:@(axis)];
  }
  return result;
}

namespace {

Tensor bitand_(Graph m, Tensor a, Tensor b) {
  return [m.g bitwiseANDWithPrimaryTensor:a secondaryTensor:b name:nil];
}
Tensor bitor_(Graph m, Tensor a, Tensor b) {
  return [m.g bitwiseORWithPrimaryTensor:a secondaryTensor:b name:nil];
}
Tensor shift_right(Graph m, Tensor a, Tensor b) {
  return [m.g bitwiseRightShiftWithPrimaryTensor:a secondaryTensor:b name:nil];
}
Tensor shift_left(Graph m, Tensor a, Tensor b) {
  return [m.g bitwiseLeftShiftWithPrimaryTensor:a secondaryTensor:b name:nil];
}
Tensor low_decode(Graph m, Tensor value, DType dtype) {
  std::array<float, 256> table{};
  size_t elements = dtype == DType::Float4E2M1FN ? 16 : 256;
  for (size_t i = 0; i < elements; ++i) {
    table[i] = dtype == DType::Float4E2M1FN ? float(numeric::float4_to_double(uint8_t(i)))
                                            : numeric::float8_to_float(uint8_t(i), dtype);
  }
  NSData* data = [NSData dataWithBytes:table.data() length:elements * sizeof(float)];
  Tensor constant = [m.g constantWithData:data shape:@[ @(elements) ] dataType:MPSDataTypeFloat32];
  return m.gather(constant, m.cast(value, MPSDataTypeInt32), 0);
}
Tensor low_encode(Graph m, Tensor value, DType dtype) {
  MPSGraph* g = m.g;
  Tensor x = m.cast(value, MPSDataTypeFloat32);
  LocalFunction u{[&](uint32_t n) { return m.number(n, MPSDataTypeUInt32); }};
  Tensor bits = [g reinterpretCastTensor:x toType:MPSDataTypeUInt32 name:nil];
  if (dtype == DType::Float4E2M1FN) {
    Tensor abs = [g absoluteWithTensor:x name:nil];
    Tensor code = m.integer(0);
    constexpr std::array<double, 7> midpoints{{.25, .75, 1.25, 1.75, 2.5, 3.5, 5.}};
    for (int i = 0; i < int(midpoints.size()); ++i) {
      Tensor midpoint = m.number(midpoints[i]);
      Tensor above = i % 2 ? [g greaterThanOrEqualToWithPrimaryTensor:abs secondaryTensor:midpoint name:nil]
                         : m.less(midpoint, abs);
      code = m.select(above, m.integer(i + 1), code);
    }
    Tensor sign = m.cast(bitand_(m, shift_right(m, bits, u(28)), u(8)), MPSDataTypeInt32);
    return bitor_(m, code, sign);
  }
  Tensor fraction = bitand_(m, bits, u(0x7fffff));
  Tensor exponent = bitand_(m, shift_right(m, bits, u(23)), u(255));
  Tensor is_nonfinite = m.equal(exponent, u(255));
  if (dtype == DType::Float8E8M0FNU) {
    Tensor round_up = [g logicalORWithPrimaryTensor:m.less(u(0x400000), fraction)
                                  secondaryTensor:m.both(m.equal(fraction, u(0x400000)), m.truth(exponent))
                                             name:nil];
    return m.cast(m.select(is_nonfinite, u(255), m.add(exponent, m.cast(round_up, MPSDataTypeUInt32))),
                  MPSDataTypeUInt8);
  }
  const bool e4 = dtype == DType::Float8E4M3FN || dtype == DType::Float8E4M3FNUZ;
  const bool unsigned_zero = dtype == DType::Float8E4M3FNUZ || dtype == DType::Float8E5M2FNUZ;
  const int mantissa = e4 ? 3 : 2, bias = (e4 ? 7 : 15) + unsigned_zero;
  Tensor sign = bitand_(m, shift_right(m, bits, u(24)), u(128));
  Tensor adjusted = m.add(m.cast(exponent, MPSDataTypeInt32), m.integer(bias - 127));
  LocalFunction round_nearest{[&](Tensor word, Tensor shift) {
    Tensor base = shift_right(m, word, shift);
    Tensor half = shift_left(m, u(1), m.sub(shift, u(1)));
    Tensor rest = bitand_(m, word, m.sub(shift_left(m, u(1), shift), u(1)));
    Tensor round_up = [g logicalORWithPrimaryTensor:m.less(half, rest)
                                  secondaryTensor:m.both(m.equal(rest, half), m.truth(bitand_(m, base, u(1))))
                                             name:nil];
    return m.add(base, m.cast(round_up, MPSDataTypeUInt32));
  }};
  Tensor sub_shift = m.sub(m.integer(24 - mantissa), adjusted);
  Tensor bounded_shift = [g clampWithTensor:sub_shift
                           minValueTensor:m.integer(1)
                           maxValueTensor:m.integer(31)
                                     name:nil];
  Tensor subnormal = round_nearest(bitor_(m, fraction, u(0x800000)), m.cast(bounded_shift, MPSDataTypeUInt32));
  subnormal = m.select(m.less(sub_shift, m.integer(32)), subnormal, u(0));
  Tensor normal = m.add(shift_left(m, m.cast(adjusted, MPSDataTypeUInt32), u(mantissa)),
                      round_nearest(fraction, u(23 - mantissa)));
  Tensor code = m.select(m.less(m.integer(0), adjusted), normal, subnormal);
  if (unsigned_zero) {
    code = m.select([g logicalORWithPrimaryTensor:is_nonfinite secondaryTensor:m.less(u(127), code) name:nil],
                    u(128), m.select(m.truth(code), bitor_(m, sign, code), u(0)));
  } else {
    Tensor positive = [g minimumWithPrimaryTensor:code secondaryTensor:u(e4 ? 127 : 124) name:nil];
    Tensor special = e4 ? u(127) : m.select(m.truth(fraction), u(127), u(124));
    code = bitor_(m, sign, m.select(is_nonfinite, special, positive));
  }
  return m.cast(code, MPSDataTypeUInt8);
}

Tensor unpack(Graph m, Tensor bytes, Shape dims, unsigned width) {
  int count = 1;
  for (NSNumber* dim in dims) {
    count *= dim.intValue;
  }
  Tensor position = m.index(@[ @(count) ], 0);
  if (8 % width) {
    LocalFunction num{[&](int value) { return m.number(value, MPSDataTypeInt64); }};
    Tensor bit = m.mul(m.cast(position, MPSDataTypeInt64), num(width));
    Tensor offset = m.cast([m.g moduloWithPrimaryTensor:bit secondaryTensor:num(8) name:nil], MPSDataTypeInt32);
    Tensor index = m.cast(m.div(bit, num(8)), MPSDataTypeInt32);
    Tensor padded = [m.g concatTensors:@[ bytes, m.zeros(@[ @1 ], MPSDataTypeUInt8) ] dimension:0 name:nil];
    Tensor low = m.cast(m.gather(padded, index, 0), MPSDataTypeInt32);
    Tensor high = m.cast(m.gather(padded, m.add(index, m.integer(1)), 0), MPSDataTypeInt32);
    Tensor word = bitor_(m, low, shift_left(m, high, m.integer(8)));
    Tensor codes = bitand_(m, shift_right(m, word, offset), m.integer((1u << width) - 1));
    return m.reshape(codes, dims);
  }
  Tensor per_byte = m.integer(8 / width);
  Tensor part = [m.g moduloWithPrimaryTensor:position secondaryTensor:per_byte name:nil];
  Tensor word = m.cast(m.gather(bytes, m.div(position, per_byte), 0), MPSDataTypeInt32);
  Tensor codes = bitand_(m, shift_right(m, word, m.mul(part, m.integer(width))), m.integer((1u << width) - 1));
  return m.reshape(codes, dims);
}
Tensor pack(Graph m, Tensor values, int count, unsigned width) {
  Tensor words = m.reshape(m.cast(values, MPSDataTypeInt32), @[ @(count) ]);
  if (8 % width) {
    NSArray<NSNumber*>* const out = @[ @((size_t(count) * width + 7) / 8) ];
    LocalFunction num{[&](int value) { return m.number(value, MPSDataTypeInt64); }};
    Tensor bit = m.mul(m.cast(m.index(out, 0), MPSDataTypeInt64), num(8));
    Tensor start = m.cast(m.div(bit, num(width)), MPSDataTypeInt32);
    Tensor offset =
        m.cast([m.g moduloWithPrimaryTensor:bit secondaryTensor:num(width) name:nil], MPSDataTypeInt32);
    Tensor padded = [m.g concatTensors:@[ words, m.zeros(@[ @1 ], MPSDataTypeInt32) ] dimension:0 name:nil];
    Tensor combined = m.integer(0);
    for (unsigned part = 0; part < (6 + 2 * width) / width; ++part) {
      Tensor index = [m.g minimumWithPrimaryTensor:m.add(start, m.integer(part))
                                 secondaryTensor:m.integer(count)
                                            name:nil];
      Tensor code = bitand_(m, m.gather(padded, index, 0), m.integer((1u << width) - 1));
      combined = bitor_(m, combined, shift_left(m, code, m.integer(part * width)));
    }
    return m.cast(bitand_(m, shift_right(m, combined, offset), m.integer(255)), MPSDataTypeUInt8);
  }
  const int parts = 8 / width, bytes = (count + parts - 1) / parts;
  if (bytes * parts != count) {
    Tensor pad = m.zeros(@[ @(bytes * parts - count) ], MPSDataTypeInt32);
    words = [m.g concatTensors:@[ words, pad ] dimension:0 name:nil];
  }
  NSArray<NSNumber*>* const expanded = @[ @(bytes), @(parts) ];
  Tensor codes = bitand_(m, m.reshape(words, expanded), m.integer((1u << width) - 1));
  Tensor shifted = shift_left(m, codes, m.mul(m.index(expanded, 1), m.integer(width)));
  return m.cast(m.sum(shifted, @[ @1 ]), MPSDataTypeUInt8);
}

// MPSGraph can move IEEE binary64 words without converting the value. Keep
// the physical lane dimension outside the logical shape used by the kernal.
Tensor wide_layout(Graph m, Lowering kernel, const std::string& op, const Operands& tensors,
                   const KernelParams& p) {
  const DType out = DType(p.output_dtype());
  NSMutableArray<MPSGraphTensor*>* lanes = [NSMutableArray<MPSGraphTensor*> arrayWithCapacity:wide_words(out)];
  for (int lane = 0; lane < wide_words(out); ++lane) {
    Operands logical = tensors;
    for (int i = 0; i < 4; ++i) {
      if (logical[i] && DType(p.input_dtype(KernelInput(i))) == out) {
        logical[i] = m.reshape(m.slice(tensors[i], 1, lane, 1), shape(p, i));
      }
    }
    Tensor result = !logical[0] && op != "concat" && op != "where" ? m.zeros(shape(p), MPSDataTypeInt32)
                                                                   : kernel(m, logical, p);
    if (!result) {
      return nil;
    }
    [lanes addObject:m.reshape(result, @[ @(p.output_count()), @1 ])];
  }
  return [m.g concatTensors:lanes dimension:1 name:nil];
}

Lowering find_kernel(const std::string& op) {
  static const std::unordered_map<std::string, Lowering> kernels = {
#define ELMAN_METAL_KERNEL(name) {#name, lower_##name},
#include "Kernals/metal_dispatch.inc"
#undef ELMAN_METAL_KERNEL
  };
  const auto it = kernels.find(op);
  return it == kernels.end() ? nullptr : it->second;
}
}  // namespace

MPSGraphTensor* lower(MPSGraph* graph, const std::string& op, const Operands& tensors, const KernelParams& p,
                      MPSGraphTensor* random_samples) {
  Graph m{graph, random_samples};
  const Lowering kernel = find_kernel(op);
  if (!kernel) {
    return nil;
  }
  if (wide_words(DType(p.output_dtype()))) {
    if (op == "cast" || op == "real" || op == "imag" || op == "complex" || op == "conj") {
      return kernel(m, tensors, p);
    }
    return wide_layout(m, kernel, op, tensors, p);
  }
  Operands logical = tensors;
  const bool bit_preserving = raw_layout(op);
  for (int i = 0; i < 4; ++i) {
    if (!logical[i]) {
      continue;
    }
    const DType dtype = DType(p.input_dtype(static_cast<KernelInput>(i)));
    const unsigned bits = packed_width(dtype);
    if (bits) {
      logical[i] = unpack(m, logical[i], shape(p, i), bits);
    }
    if (is_low_precision_float(dtype) && !bit_preserving) {
      logical[i] = low_decode(m, logical[i], dtype);
    }
    if (bits && !bit_preserving && is_packed_integer(dtype) &&
        dtype_info(dtype).kind == DTypeKind::SignedInteger) {
      Tensor sign = m.integer(1u << (bits - 1));
      logical[i] = m.sub([graph bitwiseXORWithPrimaryTensor:logical[i] secondaryTensor:sign name:nil], sign);
    }
  }
  Tensor result = kernel(m, logical, p);
  if (!result) {
    return nil;
  }
  const DType out = DType(p.output_dtype());
  if (is_low_precision_float(out) && !bit_preserving) {
    result = low_encode(m, result, out);
  }
  const unsigned int bits = packed_width(out);
  return bits ? pack(m, result, p.output_count(), bits) : result;
}
}  // namespace elman::metal
