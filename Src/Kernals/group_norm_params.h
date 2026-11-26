#pragma once
#include "Core/Backend.h"
#include <cstring>

namespace elman {
// CPU group normalization uses a double-precision epsilon for statistics.
// Keep the legacy Float32 slot for kernels that convert epsilon to opmath.
inline std::vector<int> group_norm_attributes(int groups, double epsilon) {
    float narrow = float(epsilon);
    int legacy;
    uint64_t wide;
    std::memcpy(&legacy, &narrow, sizeof(legacy));
    std::memcpy(&wide, &epsilon, sizeof(wide));
    return {groups, legacy, 1, int(uint32_t(wide)), int(uint32_t(wide >> 32))};
}
inline double group_norm_epsilon(const KernelParams& p) {
    if (p.attribute(2)) {
        uint64_t bits = uint64_t(uint32_t(p.attribute(3))) | (uint64_t(uint32_t(p.attribute(4))) << 32);
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    int bits = p.attribute(1);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
} // namespace elman
