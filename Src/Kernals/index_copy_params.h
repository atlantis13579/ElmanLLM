#pragma once
#include "Core/Backend.h"

namespace elman {
#ifdef __CUDACC__
__host__ __device__
#endif
inline int64_t index_copy_index(const uint32_t* data, int position, int dtype) {
    return dtype == int(DType::Int64) ? reinterpret_cast<const int64_t*>(data)[position] :
                                      int64_t(reinterpret_cast<const int32_t*>(data)[position]);
}
} // namespace elman
