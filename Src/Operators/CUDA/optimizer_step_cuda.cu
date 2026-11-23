#include "kernel.cuh"
#include "Operators/low_precision_optimizer_math.h"

namespace elman {
namespace {
__global__ void optimizer_step_kernel(uint32_t* output,const uint32_t* weights,const uint32_t* gradients,const uint32_t* first,const uint32_t* second,KernelParams p,FloatOptimizerParameters scalars,unsigned writers){
    const unsigned writer=blockIdx.x*blockDim.x+threadIdx.x;if(writer>=writers)return;
    const int mode=p.attribute(optimizer_attr::kResult);const DType dtype=DType(p.output_dtype());
    const bool packed=mode==0&&dtype==DType::Float4E2M1FN;const int i=packed?int(writer*2):int(writer);
    const float value=optimizer_result(i,weights,gradients,first,second,p,scalars);
    if(mode!=0||dtype==DType::Float32)reinterpret_cast<float*>(output)[i]=value;
    else{uint8_t code=low_precision_encode(value,dtype);
        if(packed&&i+1<p.output_count())code|=uint8_t(low_precision_encode(optimizer_result(i+1,weights,gradients,first,second,p,scalars),dtype)<<4);
        reinterpret_cast<uint8_t*>(output)[writer]=code;}
}
}
void launch_optimizer_step_cuda(uint32_t* output,const uint32_t* weights,const uint32_t* gradients,const uint32_t* first,const uint32_t* second,const KernelParams& p){
    const unsigned count=unsigned(p.output_count());
    const unsigned writers=p.attribute(optimizer_attr::kResult)==0&&p.output_dtype()==int(DType::Float4E2M1FN)?(count+1u)/2u:count;
    if(writers)optimizer_step_kernel<<<(writers+255u)/256u,256,0,current_cuda_stream()>>>(output,weights,gradients,first,second,p,float_optimizer_parameters(p),writers);
}
} // namespace elman
