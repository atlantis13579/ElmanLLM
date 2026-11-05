#pragma once

namespace elman::tensor_axis {

namespace image {
inline constexpr int kBatch = 0;
inline constexpr int kChannels = 1;
inline constexpr int kHeight = 2;
inline constexpr int kWidth = 3;
}  // namespace image

namespace logits {
inline constexpr int kBatch = 0;
inline constexpr int kClasses = 1;
}  // namespace logits

}  // namespace elman::tensor_axis
