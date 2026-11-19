#include "kernel.cuh"
#include "Operators/low_precision_optimizer_math.h"

namespace elman {
namespace {
__device__ float resident_float32_element(int index, const uint32_t* weights, const uint32_t* gradients,
                                          const uint32_t* first, const uint32_t* second, const KernelParams& p, int mode) {
    const int algorithm = p.attribute(optimizer_attr::kAlgorithm);
    const double rate = optimizer_parameter(p, 0), scale = optimizer_parameter(p, 1);
    const double decay = optimizer_parameter(p, 2), momentum = optimizer_parameter(p, 3);
    const double parameter = reinterpret_cast<const float*>(weights)[index];
    const double old_first = reinterpret_cast<const float*>(first)[index];
    const double old_second = reinterpret_cast<const float*>(second)[index];
    const double gradient = __dadd_rn(__dmul_rn(reinterpret_cast<const float*>(gradients)[index], scale),
                                      algorithm == 2 ? 0. : __dmul_rn(decay, parameter));
    if (algorithm == 0) {
        const float moment = static_cast<float>(__dadd_rn(__dmul_rn(momentum, old_first), gradient));
        if (mode == 1) return moment;
        if (mode == 2) return static_cast<float>(old_second);
        return static_cast<float>(__dsub_rn(parameter, __dmul_rn(rate, moment)));
    }
    const double beta1 = optimizer_parameter(p, 4), beta2 = optimizer_parameter(p, 5);
    const double epsilon = optimizer_parameter(p, 6), correction1 = optimizer_parameter(p, 7), correction2 = optimizer_parameter(p, 8);
    const float m = static_cast<float>(__dadd_rn(__dmul_rn(beta1, old_first), __dmul_rn(1. - beta1, gradient)));
    const float v = static_cast<float>(__dadd_rn(__dmul_rn(beta2, old_second),
                                                __dmul_rn(__dmul_rn(1. - beta2, gradient), gradient)));
    if (mode == 1) return m;
    if (mode == 2) return v;
    double update = (m / correction1) / (sqrt(v / correction2) + epsilon);
    if (algorithm == 2) update = __dadd_rn(update, __dmul_rn(decay, parameter));
    return static_cast<float>(__dsub_rn(parameter, __dmul_rn(rate, update)));
}

__global__ void optimizer_step_kernel(uint32_t* output,const uint32_t* weights,const uint32_t* gradients,const uint32_t* first,const uint32_t* second,KernelParams p,unsigned writers){
    const unsigned writer=blockIdx.x*blockDim.x+threadIdx.x;if(writer>=writers)return;
    const int mode=p.attribute(optimizer_attr::kResult);const auto dtype=DType(p.output_dtype());
    if (p.input_dtype(KernelInput::A) == int(DType::Float32)) {
        reinterpret_cast<float*>(output)[writer] = resident_float32_element(int(writer), weights, gradients, first, second, p, mode);
        return;
    }
    const bool packed=mode==0&&dtype==DType::Float4E2M1FN;const int i=packed?int(writer*2):int(writer);
    const float value=low_optimizer_result(low_optimizer_element(i,weights,gradients,first,second,p),mode);
    if(mode!=0)reinterpret_cast<float*>(output)[i]=value;
    else{uint8_t code=low_precision_encode(value,dtype);
        if(packed&&i+1<p.output_count())code|=uint8_t(low_precision_encode(low_optimizer_element(i+1,weights,gradients,first,second,p).value,dtype)<<4);
        reinterpret_cast<uint8_t*>(output)[writer]=code;}
}
}
void launch_optimizer_step_cuda(uint32_t* output,const uint32_t* weights,const uint32_t* gradients,const uint32_t* first,const uint32_t* second,const KernelParams& p){
    const unsigned count=unsigned(p.output_count());
    const unsigned writers=p.attribute(optimizer_attr::kResult)==0&&p.output_dtype()==int(DType::Float4E2M1FN)?(count+1u)/2u:count;
    if(writers)optimizer_step_kernel<<<(writers+255u)/256u,256,0,current_cuda_stream()>>>(output,weights,gradients,first,second,p,writers);
}
} // namespace elman
