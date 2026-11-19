#pragma once
#include "Core/Backend.h"

#ifdef __CUDACC__
#define ELMAN_OPTIMIZER_HD __host__ __device__
#else
#define ELMAN_OPTIMIZER_HD
#endif
namespace elman {
namespace optimizer_attr {
constexpr int kAlgorithm=0,kResult=1,kFirstStep=2,kResidentFloat32=3,kScalars=4,kScalarCount=9,kUsedAttributes=22;
}
ELMAN_OPTIMIZER_HD inline double optimizer_parameter(const KernelParams& p,int index){
    union {uint64_t bits;double value;} scalar;
    const int offset=optimizer_attr::kScalars+2*index;
    scalar.bits=uint64_t(uint32_t(p.attribute(offset)))|(uint64_t(uint32_t(p.attribute(offset+1)))<<32);
    return scalar.value;
}
void validate_low_precision_optimizer(const BufferPtr& output,const std::vector<BufferPtr>& inputs,const KernelParams& params);
} // namespace elman
#undef ELMAN_OPTIMIZER_HD
