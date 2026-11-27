#pragma once

namespace elman::kernal_attr::conv2d {

inline constexpr int kBatchAxis = 0;
inline constexpr int kOutputChannelAxis = 0;
inline constexpr int kChannelAxis = 1;
inline constexpr int kHeightAxis = 2;
inline constexpr int kWidthAxis = 3;

inline constexpr int kStrideHeight = 0;
inline constexpr int kStrideWidth = 1;
inline constexpr int kPaddingHeight = 2;
inline constexpr int kPaddingWidth = 3;
inline constexpr int kDilationHeight = 4;
inline constexpr int kDilationWidth = 5;
inline constexpr int kGroups = 6;

}  // namespace elman::kernal_attr::conv2d
