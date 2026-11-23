#include <climits>
#include <cstring>

#include "Backends/MPSGraph.h"
#include "Core/LocalFunction.h"

#include "Operators/attributes.h"

namespace elman::metal {
namespace {
Tensor causal_mask(Graph m, Tensor mask, int queries, int keys, int64_t offset, MPSDataType type) {
  if (offset >= keys - 1) {
    return mask;
  }
  const float lowest = type == MPSDataTypeFloat16    ? -65504.f
                       : type == MPSDataTypeBFloat16 ? -3.3895313892515355e38f
                                                     : -3.402823466e38f;
  Tensor blocked = m.number(lowest, mask.dataType);
  if (offset <= -int64_t(queries)) {
    return blocked;
  }
  NSArray<NSNumber*>* dims = @[ @1, @1, @(queries), @(keys) ];
  Tensor distance = m.sub(m.index(dims, 3), m.index(dims, 2));
  return m.select(m.less(m.integer(offset), distance), blocked, mask);
}
}  // namespace

Tensor lower_paged_attention(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor q = inputs[0], k = inputs[1], v = inputs[2], mask = inputs[3];
  const int batch = p.input_shape(KernelInput::A, 0);
  const int dimension = p.input_shape(KernelInput::A, 3);
  const int key_count = p.input_shape(KernelInput::D, 1), kv_heads = p.attribute(4);
  if (key_count == 0) {
    return m.zeros(shape(p), q.dataType);
  }
  const float minimum = q.dataType == MPSDataTypeFloat16    ? -65504.f
                        : q.dataType == MPSDataTypeBFloat16 ? -3.3895313892515355e38f
                                                            : -3.402823466e38f;
  Tensor inactive = m.less(mask, m.integer(0));
  Tensor unbound = m.equal(mask, m.integer(INT_MIN));
  Tensor indices = m.select(inactive, m.sub(m.integer(-1), mask), mask);
  indices = m.select(unbound, m.integer(0), indices);
  LocalFunction lookup{[&](Tensor x) {
    Tensor data = m.gather(x, indices, 0);
    data = m.select(m.reshape(unbound, @[ @(batch), @(key_count), @1 ]), m.number(0, x.dataType), data);
    return [m.g transposeTensor:m.reshape(data, @[ @(batch), @(key_count), @(kv_heads), @(dimension) ])
                    permutation:@[ @0, @2, @1, @3 ]
                           name:nil];
  }};
  k = lookup(k);
  v = lookup(v);
  mask = m.reshape(m.select(inactive, m.number(minimum, q.dataType), m.number(0, q.dataType)),
                   @[ @(batch), @1, @1, @(key_count) ]);
  const int heads = p.input_shape(KernelInput::A, 1);
  const int queries = p.input_shape(KernelInput::A, 2);
  const int value_width = p.output_shape(3);
  const bool causal = p.attribute(operator_attr::tiled_attention::kCausal) != 0;
  const int causal_offset = p.attribute(operator_attr::tiled_attention::kCausalOffset);
  const bool fused = dimension == value_width && dimension % 8 == 0 && !(causal && causal_offset < 0);
  const bool grouped_decode = fused && !p.attribute(5) && queries == 1 && heads != kv_heads &&
                              dimension == 128 && key_count >= 256 &&
                              (q.dataType == MPSDataTypeFloat16 || q.dataType == MPSDataTypeBFloat16);
  if (grouped_decode) {
    // Query heads in a group share the same keys, values, and single-position mask.
    q = m.reshape(q, @[ @(batch), @(kv_heads), @(heads / kv_heads), @(dimension) ]);
  } else if (heads != kv_heads) {
    NSMutableArray<NSNumber*>* repeat = [NSMutableArray<NSNumber*> arrayWithCapacity:heads];
    for (int i = 0; i < heads; ++i) {
      [repeat addObject:@(i / (heads / kv_heads))];
    }
    Tensor ids = m.int_vector(repeat);
    k = m.gather(k, ids, 1);
    v = m.gather(v, ids, 1);
  }
  if (causal) {
    mask = causal_mask(m, mask, queries, key_count, causal_offset, q.dataType);
  }
  float divisor = attribute_float(p, 0);
  if (p.attribute(7)) {
    const uint64_t bits = uint64_t(uint32_t(p.attribute(5))) | (uint64_t(uint32_t(p.attribute(6))) << 32);
    double encoded;
    std::memcpy(&encoded, &bits, sizeof(encoded));
    divisor = float(encoded);
  }
  const float scale = 1.f / divisor;
  if (!fused) {
    // A finite additive mask on a fully blocked row still participates in
    // softmax. Fused SDPA is allowed to suppress such rows entirely.
    Tensor logits = m.matmul(q, m.transpose(k, 2, 3));
    logits = m.add(m.mul(logits, m.number(scale, logits.dataType)), mask);
    return m.matmul([m.g softMaxWithTensor:logits axis:3 name:nil], v);
  }
  Tensor result = [m.g scaledDotProductAttentionWithQueryTensor:q
                                                    keyTensor:k
                                                  valueTensor:v
                                                   maskTensor:mask
                                                        scale:scale
                                                         name:nil];
  return grouped_decode ? m.reshape(result, shape(p)) : result;
}
}  // namespace elman::metal
