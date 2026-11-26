#include "kernel.cuh"
#include <math_constants.h>
#include "Kernals/sdpa.h"

namespace elman {
namespace {
template<int Items,bool Backward>
__global__ void sdpa_rows(float* output,const float* a,const uint32_t* b,KernelParams p,int rows,int width) {
    const int row=int(blockIdx.x)*4+int(threadIdx.y), lane=int(threadIdx.x);
    const int rank=p.input_rank(KernelInput::A), queries=p.input_shape(KernelInput::A,rank-2);
    const float scale=__int_as_float(p.attribute(sdpa_attr::scale));
    const float threshold=__int_as_float(p.attribute(sdpa_attr::threshold)), multiplier=__int_as_float(p.attribute(sdpa_attr::dropout_scale));
    float values[Items], gradient[Items], total=0, maximum=-CUDART_INF_F;
    bool has_nan=false;
    #pragma unroll
    for(int j=0;j<Items;++j) {
        const int column=lane+j*32, index=row*width+column;
        const bool valid=row<rows && column<width;
        float x=valid?a[index]:0.f;
        const float drop=elman_random(index,p.attribute(sdpa_attr::seed),0)>=threshold?multiplier:0.f;
        if constexpr(Backward) {
            gradient[j]=valid?reinterpret_cast<const float*>(b)[index]*drop:0.f;
            total=fmaf(x,gradient[j],total);
        } else {
            x=valid?__fmul_rn(x,scale):-CUDART_INF_F;
            if(valid && p.input_rank(KernelInput::B)>=0) {
                int offset=0, remaining=index;
                for(int d=rank-1;d>=0;--d) {
                    int c=remaining%p.input_shape(KernelInput::A,d); remaining/=p.input_shape(KernelInput::A,d);
                    int m=d-rank+p.input_rank(KernelInput::B);
                    if(m>=0 && p.input_shape(KernelInput::B,m)!=1) offset+=c*p.input_stride(KernelInput::B,m);
                }
                if(p.input_dtype(KernelInput::B)==int(DType::Bool)) { if(!reinterpret_cast<const uint8_t*>(b)[offset]) x=-CUDART_INF_F; }
                else x=__fadd_rn(x,reinterpret_cast<const float*>(b)[offset]);
            }
            if(p.attribute(sdpa_attr::causal) && column>row%queries) x=-CUDART_INF_F;
            has_nan=has_nan || isnan(x);
            maximum=fmaxf(maximum,x);
        }
        values[j]=x;
    }
    if constexpr(!Backward) {
        for(int offset=16;offset;offset/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
        if(__any_sync(0xffffffff,has_nan)) maximum=CUDART_NAN_F;
        #pragma unroll
        for(int j=0;j<Items;++j) { values[j]=maximum==-CUDART_INF_F?0.f:expf(values[j]-maximum); total+=values[j]; }
    }
    for(int offset=16;offset;offset/=2) total+=__shfl_xor_sync(0xffffffff,total,offset);
    #pragma unroll
    for(int j=0;j<Items;++j) {
        const int column=lane+j*32,index=row*width+column;
        if(row<rows && column<width) {
            if constexpr(Backward) output[index]=__fmul_rn(__fmul_rn(values[j],__fsub_rn(gradient[j],total)),scale);
            else {
                const float probability=total==0.f?0.f:values[j]/total;
                const float drop=elman_random(index,p.attribute(sdpa_attr::seed),0)>=threshold?multiplier:0.f;
                output[index]=probability; output[rows*width+index]=probability*drop;
            }
        }
    }
}
template<bool Backward>
void launch(uint32_t* output,const uint32_t* a,const uint32_t* b,const KernelParams& p) {
    const int width=p.input_shape(KernelInput::A,p.input_rank(KernelInput::A)-1);
    const int count=Backward?p.output_count():p.output_count()/2, rows=count/width;
    if(!rows) return;
    int power=32; while(power<width) power*=2;
    #define ELMAN_SDPA_CASE(n) case n: sdpa_rows<n/32,Backward><<<(rows+3)/4,dim3(32,4),0,current_cuda_stream()>>>(reinterpret_cast<float*>(output),reinterpret_cast<const float*>(a),b,p,rows,width); break;
    switch(power) { ELMAN_SDPA_CASE(32) ELMAN_SDPA_CASE(64) ELMAN_SDPA_CASE(128) ELMAN_SDPA_CASE(256) ELMAN_SDPA_CASE(512) ELMAN_SDPA_CASE(1024) }
    #undef ELMAN_SDPA_CASE
}
}
void launch_sdpa_weights_cuda(uint32_t* output,const uint32_t* a,const uint32_t* b,const uint32_t*,const KernelParams& p) { launch<false>(output,a,b,p); }
void launch_sdpa_gradient_cuda(uint32_t* output,const uint32_t* a,const uint32_t* b,const uint32_t*,const KernelParams& p) { launch<true>(output,a,b,p); }
}
