#pragma once
#include "Operators/low_precision_optimizer.h"
#include "Operators/low_precision_pointwise_math.h"

#ifdef __CUDACC__
#define ELMAN_LOW_OPTIMIZER_MATH __device__
#else
#define ELMAN_LOW_OPTIMIZER_MATH
#endif
namespace elman {
ELMAN_LOW_OPTIMIZER_MATH inline float optimizer_mul(float a,float b){
#ifdef __CUDA_ARCH__
    return __fmul_rn(a,b);
#else
    volatile float result=a*b;return result;
#endif
}
ELMAN_LOW_OPTIMIZER_MATH inline float optimizer_add(float a,float b){
#ifdef __CUDA_ARCH__
    return __fadd_rn(a,b);
#else
    volatile float result=a+b;return result;
#endif
}
ELMAN_LOW_OPTIMIZER_MATH inline float optimizer_sub(float a,float b){
#ifdef __CUDA_ARCH__
    return __fsub_rn(a,b);
#else
    volatile float result=a-b;return result;
#endif
}
ELMAN_LOW_OPTIMIZER_MATH inline float optimizer_div(float a,float b){
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a,b);
#else
    volatile float result=a/b;return result;
#endif
}
ELMAN_LOW_OPTIMIZER_MATH inline float optimizer_scaled_add(float a,float b,float scale,int index,int count){
#ifdef __CUDA_ARCH__
    return ::fmaf(scale,b,a);
#else
    // CPU TensorIterator uses two AVX2 vectors before its scalar tail.
    return index<count/16*16?::fmaf(scale,b,a):optimizer_add(a,optimizer_mul(scale,b));
#endif
}
struct LowOptimizerElement {float value,first,second;};
ELMAN_LOW_OPTIMIZER_MATH inline LowOptimizerElement low_optimizer_element(int index,const uint32_t* weights,const uint32_t* gradients,
        const uint32_t* first,const uint32_t* second,const KernelParams& p){
    const auto dtype=DType(p.input_dtype(KernelInput::A));
    const double rate=optimizer_parameter(p,0),scale=optimizer_parameter(p,1),decay=optimizer_parameter(p,2),momentum=optimizer_parameter(p,3),
        beta1=optimizer_parameter(p,4),beta2=optimizer_parameter(p,5),epsilon=optimizer_parameter(p,6),correction1=optimizer_parameter(p,7),correction2_sqrt=optimizer_parameter(p,8);
    const int algorithm=p.attribute(optimizer_attr::kAlgorithm);
    const float parameter=low_precision_value(weights,index,dtype);
    float g=optimizer_mul(low_precision_value(gradients,index,dtype),float(scale));
    if(algorithm!=2&&decay!=0.)g=optimizer_scaled_add(g,parameter,float(decay),index,p.output_count());
    const float old_first=reinterpret_cast<const float*>(first)[index],old_second=reinterpret_cast<const float*>(second)[index];
    LowOptimizerElement result;
    if(algorithm==0){
        result.first=momentum==0.||p.attribute(optimizer_attr::kFirstStep)?g:optimizer_add(optimizer_mul(old_first,float(momentum)),g);
        result.second=old_second;
        result.value=optimizer_scaled_add(parameter,result.first,float(-rate),index,p.output_count());return result;
    }
    const float weight=float(1.-beta1),delta=optimizer_sub(g,old_first);
#ifdef __CUDA_ARCH__
    result.first=weight<.5f ? ::fmaf(weight,delta,old_first) : ::fmaf(-delta,optimizer_sub(1.f,weight),g);
    result.second=::fmaf(float(1.-beta2),optimizer_mul(g,g),optimizer_mul(old_second,float(beta2)));
#else
    if(index<p.output_count()/16*16)result.first=::fmaf(weight<.5f?weight:optimizer_sub(weight,1.f),delta,weight<.5f?old_first:g);
    else result.first=weight<.5f?optimizer_add(old_first,optimizer_mul(weight,delta)):optimizer_sub(g,optimizer_mul(delta,optimizer_sub(1.f,weight)));
    result.second=optimizer_add(optimizer_mul(old_second,float(beta2)),optimizer_mul(optimizer_mul(float(1.-beta2),g),g));
#endif
    const float root=::sqrtf(result.second);
#ifdef __CUDA_ARCH__
    const float denominator=optimizer_add(optimizer_mul(root,optimizer_div(1.f,float(correction2_sqrt))),float(epsilon));
#else
    const float denominator=optimizer_add(optimizer_div(root,float(correction2_sqrt)),float(epsilon));
#endif
    const float value=algorithm==2&&decay!=0.?optimizer_mul(parameter,float(1.-rate*decay)):parameter;
#ifdef __CUDA_ARCH__
    result.value=::fmaf(float(-rate/correction1),optimizer_div(result.first,denominator),value);
#else
    result.value=optimizer_add(value,optimizer_div(optimizer_mul(float(-rate/correction1),result.first),denominator));
#endif
    return result;
}
ELMAN_LOW_OPTIMIZER_MATH inline float low_optimizer_result(const LowOptimizerElement& element,int mode){return mode==1?element.first:mode==2?element.second:element.value;}
} // namespace elman
#undef ELMAN_LOW_OPTIMIZER_MATH
