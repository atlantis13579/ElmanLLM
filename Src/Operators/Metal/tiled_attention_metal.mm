#include <cstring>

#include "Backends/MPSGraph.h"

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

Tensor lower_tiled_attention(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor query = inputs[0];
  Tensor key = inputs[1];
  Tensor value = inputs[2];
  Tensor mask = inputs[3];
  const int key_count = p.input_shape(KernelInput::B, 2);
  if (!key_count) {
    return m.zeros(shape(p), query.dataType);
  }
  const int kv_heads = p.input_shape(KernelInput::B, 1);
  const int heads = p.input_shape(KernelInput::A, 1);
  const int queries = p.input_shape(KernelInput::A, 2);
  const int dimension = p.input_shape(KernelInput::A, 3);
  if (heads != kv_heads) {
    NSMutableArray<NSNumber*>* repeat = [NSMutableArray<NSNumber*> arrayWithCapacity:heads];
    for (int i = 0; i < heads; ++i) {
      [repeat addObject:@(i / (heads / kv_heads))];
    }
    Tensor ids = m.int_vector(repeat);
    key = m.gather(key, ids, 1);
    value = m.gather(value, ids, 1);
  }
  if (p.attribute(operator_attr::tiled_attention::kCausal)) {
    mask = causal_mask(m, mask, queries, key_count,
                       p.attribute(operator_attr::tiled_attention::kCausalOffset), query.dataType);
  }
  float divisor = attribute_float(p, 0);
  if (p.attribute(7)) {
    const uint64_t bits = uint64_t(uint32_t(p.attribute(5))) | (uint64_t(uint32_t(p.attribute(6))) << 32);
    double encoded;
    std::memcpy(&encoded, &bits, sizeof(encoded));
    divisor = float(encoded);
  }
  const float scale = 1.f / divisor;
  const int value_width = p.output_shape(3);
  const bool fully_blocked_rows = p.attribute(operator_attr::tiled_attention::kCausal) &&
                                  p.attribute(operator_attr::tiled_attention::kCausalOffset) < 0;
  if (dimension != value_width || dimension % 8 != 0 || fully_blocked_rows) {
    // A finite additive mask on a fully blocked row still participates in
    // softmax. Fused SDPA is allowed to suppress such rows entirely.
    Tensor logits = m.matmul(query, m.transpose(key, 2, 3));
    logits = m.add(m.mul(logits, m.number(scale, logits.dataType)), mask);
    return m.matmul([m.g softMaxWithTensor:logits axis:3 name:nil], value);
  }
  return [m.g scaledDotProductAttentionWithQueryTensor:query
                                             keyTensor:key
                                           valueTensor:value
                                            maskTensor:mask
                                                 scale:scale
                                                  name:nil];
}
}  // namespace elman::metal
