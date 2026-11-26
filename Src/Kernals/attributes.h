#pragma once

namespace elman::operator_attr {

namespace addmm {
inline constexpr int kAlpha = 0;
inline constexpr int kBeta = 1;
}  // namespace addmm

namespace reduction {
inline constexpr int kAxis = 0;
inline constexpr int kKeepDimension = 1;
inline constexpr int kGradientKind = 2;
}  // namespace reduction

namespace pool2d {
inline constexpr int kKernelHeight = 0;
inline constexpr int kKernelWidth = 1;
inline constexpr int kStrideHeight = 2;
inline constexpr int kStrideWidth = 3;
inline constexpr int kPaddingHeight = 4;
inline constexpr int kPaddingWidth = 5;
inline constexpr int kCountIncludePadding = 6;
}  // namespace pool2d

namespace batch_norm {
inline constexpr int kEpsilon = 0;
}

namespace concat {
inline constexpr int kAxis = 0;
}

namespace dropout {
inline constexpr int kSeed = 0;
inline constexpr int kStreamOffset = 1;
inline constexpr int kProbability = 2;
inline constexpr int kTraining = 3;
inline constexpr int kBackward = 9;
}  // namespace dropout

namespace indexing {
inline constexpr int kAxis = 0;
}
namespace index_copy {
inline constexpr int kAxis = 0;
// Derived by backend validation, never trusted from a caller's descriptor.
inline constexpr int kHasDuplicates = 1;
}

namespace group_norm {
inline constexpr int kGroups = 0;
inline constexpr int kEpsilon = 1;
}  // namespace group_norm

namespace normalization {
inline constexpr int kTrailingDimensions = 0;
inline constexpr int kEpsilon = 1;
// RMSNorm only: 0 rounds the output; 1 also rounds before the affine weight.
inline constexpr int kRounding = 2;
}  // namespace normalization

namespace activation {
inline constexpr int kNegativeSlope = 0;
}

namespace softmax {
inline constexpr int kAxis = 0;
}

namespace attention_softmax {
inline constexpr int kAxis = 0;
inline constexpr int kDivisor = 1;
// Internal tiled-attention mask transformation; ordinary dispatch keeps these zero.
inline constexpr int kCausal = 2;
inline constexpr int kCausalOffset = 3;
inline constexpr int kQueryBase = 4;
}
namespace packed_linear {
inline constexpr int kCount = 0;
inline constexpr int kWidths = 1;
}
namespace tiled_attention {
inline constexpr int kDivisor = 0;
inline constexpr int kQueryTile = 1;
inline constexpr int kCausal = 2;
inline constexpr int kCausalOffset = 3;
inline constexpr int kTiling = 4;
}

namespace random {
inline constexpr int kSeed = 0;
inline constexpr int kStreamOffset = 1;
inline constexpr int kLowerBound = 2;
inline constexpr int kUpperBound = 3;
inline constexpr int kMean = 2;
inline constexpr int kStandardDeviation = 3;
}  // namespace random

namespace slice {
inline constexpr int kAxis = 0;
inline constexpr int kStart = 1;
inline constexpr int kStep = 2;
}  // namespace slice

namespace transpose {
inline constexpr int kFirstAxis = 0;
inline constexpr int kSecondAxis = 1;
}  // namespace transpose

namespace triangular {
inline constexpr int kDiagonal = 0;
}

namespace permutation {
inline constexpr int kFirstAxis = 0;
}

}  // namespace elman::operator_attr
