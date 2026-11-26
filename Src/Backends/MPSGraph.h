#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <array>
#include <vector>

#include "Core/Backend.h"

namespace elman::metal {
using Tensor = MPSGraphTensor*;
using Shape = MPSShape*;
using Operands = std::array<Tensor, 4>;

Shape shape(const KernelParams& params, int input = -1);
Shape storage_shape(const KernelParams& params, int input = -1);
MPSDataType storage_type(DType dtype);
Tensor lower(MPSGraph* graph, const std::string& op, const Operands& tensors, const KernelParams& params,
             Tensor random_samples);
MPSMatrixRandomDistributionDescriptor* dropout_distribution(const KernelParams& p);
MPSMatrixRandomDistributionDescriptor* normal_distribution(const KernelParams& p);
MPSMatrixRandomDistributionDescriptor* uniform_distribution(const KernelParams& p);

float attribute_float(const KernelParams& p, int slot);
NSArray<NSNumber*>* axes(int start, int end);

// Primitive MPSGraph nodes and tensor storage conversions for operator lowering.
struct Graph {
  MPSGraph* g;
  Tensor random_samples;
  Tensor number(double value, MPSDataType type = MPSDataTypeFloat32) const {
    return [g constantWithScalar:value dataType:type];
  }
  Tensor integer(int64_t value) const {
    return number(value, MPSDataTypeInt32);
  }
  Tensor cast(Tensor x, MPSDataType type) const {
    return !x || x.dataType == type ? x : [g castTensor:x toType:type name:nil];
  }
  Tensor reshape(Tensor x, Shape dims) const {
    return [g reshapeTensor:x withShape:dims name:nil];
  }
  Tensor spread(Tensor x, Shape dims) const {
    return [g broadcastTensor:x toShape:dims name:nil];
  }
  Tensor float_work(Tensor x) const {
    return x && (x.dataType == MPSDataTypeFloat16 || x.dataType == MPSDataTypeBFloat16)
               ? cast(x, MPSDataTypeFloat32)
               : x;
  }
  Tensor truth(Tensor x) const {
    return x.dataType == MPSDataTypeBool
               ? x
               : [g notEqualWithPrimaryTensor:x secondaryTensor:number(0, x.dataType) name:nil];
  }
  Tensor add(Tensor x, Tensor y) const {
    return [g additionWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor sub(Tensor x, Tensor y) const {
    return [g subtractionWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor mul(Tensor x, Tensor y) const {
    return [g multiplicationWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor div(Tensor x, Tensor y) const {
    return [g divisionWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor equal(Tensor x, Tensor y) const {
    return [g equalWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor less(Tensor x, Tensor y) const {
    return [g lessThanWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor both(Tensor x, Tensor y) const {
    // MPSGraph logicalAND requires identical operand types. Normalize both
    // operands to bool so integer comparison results never reach the op as si32.
    return [g logicalANDWithPrimaryTensor:truth(x) secondaryTensor:truth(y) name:nil];
  }
  Tensor select(Tensor predicate, Tensor yes, Tensor no) const {
    return [g selectWithPredicateTensor:predicate truePredicateTensor:yes falsePredicateTensor:no name:nil];
  }
  Tensor square(Tensor x) const {
    return [g squareWithTensor:x name:nil];
  }
  Tensor sqrt(Tensor x) const {
    return [g squareRootWithTensor:x name:nil];
  }
  Tensor rsqrt(Tensor x) const {
    return [g reciprocalSquareRootWithTensor:x name:nil];
  }
  Tensor neg(Tensor x) const {
    return [g negativeWithTensor:x name:nil];
  }
  Tensor exp(Tensor x) const {
    return [g exponentWithTensor:x name:nil];
  }
  Tensor log(Tensor x) const {
    return [g logarithmWithTensor:x name:nil];
  }
  Tensor sum(Tensor x, NSArray<NSNumber*>* dims) const {
    return [g reductionSumWithTensor:x axes:dims name:nil];
  }
  Tensor mean(Tensor x, NSArray<NSNumber*>* dims) const {
    return [g meanOfTensor:x axes:dims name:nil];
  }
  Tensor max(Tensor x, NSArray<NSNumber*>* dims) const {
    return x.dataType == MPSDataTypeFloat32 || x.dataType == MPSDataTypeFloat16 ||
                   x.dataType == MPSDataTypeBFloat16
               ? [g reductionMaximumPropagateNaNWithTensor:x axes:dims name:nil]
               : [g reductionMaximumWithTensor:x axes:dims name:nil];
  }
  Tensor min(Tensor x, NSArray<NSNumber*>* dims) const {
    return x.dataType == MPSDataTypeFloat32 || x.dataType == MPSDataTypeFloat16 ||
                   x.dataType == MPSDataTypeBFloat16
               ? [g reductionMinimumPropagateNaNWithTensor:x axes:dims name:nil]
               : [g reductionMinimumWithTensor:x axes:dims name:nil];
  }
  Tensor maximum(Tensor x, Tensor y) const {
    return [g maximumWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor minimum(Tensor x, Tensor y) const {
    return [g minimumWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor index(Shape dims, int axis) const {
    return [g coordinateAlongAxis:axis withShape:dims name:nil];
  }
  Tensor bounded(Tensor x, int size) const {
    return both([g greaterThanOrEqualToWithPrimaryTensor:x secondaryTensor:number(0, x.dataType) name:nil],
                less(x, number(size, x.dataType)));
  }
  Tensor slice(Tensor x, int axis, int start, int length) const {
    return [g sliceTensor:x dimension:axis start:start length:length name:nil];
  }
  Tensor gather(Tensor x, Tensor ids, int axis) const {
    return [g gatherWithUpdatesTensor:x indicesTensor:ids axis:axis batchDimensions:0 name:nil];
  }
  Tensor matmul(Tensor x, Tensor y) const {
    return [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:y name:nil];
  }
  Tensor transpose(Tensor x, int i, int j) const {
    return [g transposeTensor:x dimension:i withDimension:j name:nil];
  }
  Tensor zeros(Shape dims, MPSDataType type) const {
    return [g constantWithScalar:0 shape:dims dataType:type];
  }
  Tensor int_vector(NSArray<NSNumber*>* values) const {
    std::vector<int32_t> words;
    words.reserve(values.count);
    for (NSNumber* value in values) {
      words.push_back(value.intValue);
    }
    NSData* bytes = [NSData dataWithBytes:words.data() length:words.size() * sizeof(int32_t)];
    return [g constantWithData:bytes shape:@[ @(words.size()) ] dataType:MPSDataTypeInt32];
  }

  Tensor filled(const KernelParams& p, double value) const {
    const DType dtype = DType(p.output_dtype());
    return spread(number(value, is_low_precision_float(dtype) ? MPSDataTypeFloat32 : storage_type(dtype)),
                  shape(p));
  }
};
}  // namespace elman::metal
