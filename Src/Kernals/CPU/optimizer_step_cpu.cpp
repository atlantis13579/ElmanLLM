#include "kernel.h"
#include "Kernals/low_precision_optimizer_math.h"

namespace elman {
void launch_optimizer_step_cpu(uint32_t* output,const uint32_t* weights,const uint32_t* gradients,const uint32_t* first,const uint32_t* second,const KernelParams& p){
    const int mode=p.attribute(optimizer_attr::kResult);const DType dtype=DType(p.output_dtype());
    const FloatOptimizerParameters scalars = float_optimizer_parameters(p);
    const bool packed=mode==0&&dtype==DType::Float4E2M1FN;
    for(int i=0;i<p.output_count();i+=packed?2:1){
        const float value=optimizer_result(i,weights,gradients,first,second,p,scalars);
        if(mode!=0)reinterpret_cast<float*>(output)[i]=value;
        else{uint8_t code=low_precision_encode(value,dtype);
            if(packed&&i+1<p.output_count())code|=uint8_t(low_precision_encode(optimizer_result(i+1,weights,gradients,first,second,p,scalars),dtype)<<4);
            reinterpret_cast<uint8_t*>(output)[packed?i/2:i]=code;}
    }
}
} // namespace elman
