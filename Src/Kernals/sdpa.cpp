#include "Kernals/sdpa.h"
#include "Core/Autograd.h"
#include "Core/Device.h"
#include "Core/TensorDispatch.h"
#include "Fusion/Fusion.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
int bits(float value) { int result; std::memcpy(&result,&value,4); return result; }
Shape batch_shape(const Shape& a, const Shape& b) {
    Shape shape(std::max(a.size(),b.size()),1);
    for(size_t i=0;i<shape.size();++i) {
        int x=i<a.size()?a[a.size()-i-1]:1, y=i<b.size()?b[b.size()-i-1]:1;
        require(x==y || x==1 || y==1,"SDPA shapes do not broadcast");
        shape[shape.size()-i-1]=x==1?y:x;
    }
    return shape;
}
Tensor scalar(double value, const Tensor& like) {
    return like.dtype()==DType::Float64 ? Tensor::from_double_data({}, {value},like.backend()) :
        cast(Tensor::from_data({}, {float(value)},like.backend()),like.dtype());
}
Tensor sum_to(Tensor value, const Shape& shape) {
    while(value.shape().size()>shape.size()) value=sum(value,0);
    for(int d=0;d<int(shape.size());++d) if(shape[d]==1 && value.shape()[d]!=1) value=sum(value,d,true);
    return reshape(value,shape);
}
Tensor left_transposed(const Tensor& a, const Tensor& b) {
    Shape shape=batch_shape(Shape(a.shape().begin(),a.shape().end()-2),Shape(b.shape().begin(),b.shape().end()-2));
    shape.push_back(a.shape().back()); shape.push_back(b.shape().back());
    return dispatch_tensor("matmul",shape,{a,b},{0,0,1},DType::Float32);
}
Tensor repeat_heads(const Tensor& value,int heads) {
    Shape shape=value.shape(); const size_t axis=shape.size()-3;
    if(shape[axis]==heads) return value;
    require(shape.size()<8,"SDPA GQA expansion requires rank below eight");
    Shape expanded=shape; expanded.insert(expanded.begin()+axis+1,heads/shape[axis]); shape[axis]=heads;
    return reshape(expand(unsqueeze(value,int(axis)+1),expanded),shape);
}
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    const Tensor& mask, double dropout_p, bool is_causal,
                                    std::optional<double> scale, bool enable_gqa, uint32_t seed) {
    require(query.defined() && key.defined() && value.defined(),"SDPA requires Q, K and V");
    require(query.shape().size()>=2 && key.shape().size()>=2 && value.shape().size()>=2,"SDPA requires rank at least two");
    require(query.dtype()==key.dtype() && query.dtype()==value.dtype() &&
            (query.dtype()==DType::Float32 || query.dtype()==DType::Float64 || is_narrow_float(query.dtype())),"SDPA requires matching Float32, Float64, Float16 or BFloat16 inputs");
    require(query.backend()==key.backend() && query.backend()==value.backend() && query.device()==key.device() && query.device()==value.device(),"SDPA inputs must share a device");
    require(std::isfinite(dropout_p) && dropout_p>=0 && dropout_p<=1,"SDPA dropout probability must be in [0,1]");
    require(!(is_causal && mask.defined()),"SDPA causal and explicit masks are mutually exclusive");
    require(query.shape().back()>0 && query.shape().back()==key.shape().back() &&
            key.shape()[key.shape().size()-2]==value.shape()[value.shape().size()-2],"SDPA Q/K dimensions or K/V sequence lengths differ");
    if(mask.defined()) require(mask.backend()==query.backend() && mask.device()==query.device() &&
        (mask.dtype()==DType::Bool || mask.dtype()==query.dtype()),"SDPA mask must be Boolean or match the input dtype and device");
    DeviceScope owner(query.backend(),query.device());
    Tensor q=query,k=key,v=value;
    if(enable_gqa) {
        require(q.shape().size()>=3 && k.shape().size()>=3 && v.shape().size()>=3,"SDPA GQA needs a head dimension");
        const int h=q.shape()[q.shape().size()-3], kh=k.shape()[k.shape().size()-3], vh=v.shape()[v.shape().size()-3];
        require(kh>0 && kh==vh && h%kh==0,"SDPA GQA requires Hq divisible by matching K/V heads");
        k=repeat_heads(k,h); v=repeat_heads(v,h);
    }
    Shape batch=batch_shape(batch_shape(Shape(q.shape().begin(),q.shape().end()-2),Shape(k.shape().begin(),k.shape().end()-2)),
                            Shape(v.shape().begin(),v.shape().end()-2));
    const int rows=q.shape()[q.shape().size()-2], columns=k.shape()[k.shape().size()-2];
    Shape scores_shape=batch; scores_shape.push_back(rows); scores_shape.push_back(columns);
    if(mask.defined()) require(batch_shape(mask.shape(),scores_shape)==scores_shape,"SDPA mask does not broadcast to attention weights");
    const double factor=scale.value_or(1./std::sqrt(double(q.shape().back())));
    const size_t count=shape_element_count(scores_shape);
    if(q.backend()==Backend::CUDA && q.dtype()==DType::Float32 && fusion::enabled() && count && columns<=1024 && count<=size_t(INT32_MAX)/2) {
        Tensor scores=matmul_transposed(q,k);
        if(scores.shape()!=scores_shape) scores=expand(scores,scores_shape);
        float threshold=float(dropout_p);
        if(double(threshold)<dropout_p) threshold=std::nextafter(threshold,1.f);
        if(dropout_p==1) threshold=std::numeric_limits<float>::infinity();
        std::vector<int> attrs{bits(float(factor)),int(is_causal),int(seed),bits(threshold),bits(dropout_p==1?0.f:1.f/float(1-dropout_p))};
        Tensor packed=dispatch_tensor("sdpa_weights",{int(count*2)},{scores,mask},attrs);
        Tensor probability=contiguous_view(packed,scores_shape), dropped=contiguous_view(packed,scores_shape,count);
        Tensor output=matmul(dropped,v);
        attrs[sdpa_attr::dropout_scale]=bits(dropout_p==1?0.f:float(1./(1-dropout_p)));
        // Saved probabilities own the state required by the standard SDPA VJP.
        return graph::tracked(output.detach(),{q,k,v,mask},[q,k,v,mask,probability,dropped,attrs,factor](const Tensor& g) {
            if(v.requires_grad()) autograd::accumulate(v,sum_to(left_transposed(dropped,g),v.shape()));
            if(q.requires_grad() || k.requires_grad() || mask.requires_grad()) {
                Tensor dp=matmul_transposed(g,v);
                if(dp.shape()!=probability.shape()) dp=expand(dp,probability.shape());
                auto parameters=attrs;
                if(mask.requires_grad()) parameters[sdpa_attr::scale]=bits(1.f);
                Tensor ds=dispatch_tensor("sdpa_gradient",probability.shape(),{probability,dp},parameters);
                if(mask.requires_grad()) { autograd::accumulate(mask,sum_to(ds,mask.shape())); ds=mul(ds,factor); }
                if(q.requires_grad()) autograd::accumulate(q,sum_to(matmul(ds,k),q.shape()));
                if(k.requires_grad()) autograd::accumulate(k,sum_to(left_transposed(ds,q),k.shape()));
            }
        });
    }
    const DType storage=q.dtype();
    if(is_narrow_float(storage)) { q=cast(q,DType::Float32); k=cast(k,DType::Float32); v=cast(v,DType::Float32); }
    Tensor scores=mul(matmul_transposed(q,k),factor);
    if(scores.shape()!=scores_shape) scores=expand(scores,scores_shape);
    if(is_causal) {
        Tensor allowed=tril(Tensor::ones({rows,columns},q.backend(),false,DType::Bool));
        scores=where(allowed,scores,scalar(-std::numeric_limits<double>::infinity(),q));
    } else if(mask.defined()) {
        scores=mask.dtype()==DType::Bool ? where(mask,scores,scalar(-std::numeric_limits<double>::infinity(),q)) : add(scores,cast(mask,q.dtype()));
    }
    Tensor weights;
    if(columns) {
        // Fully masked rows are zero, including their derivative. Avoid NaNs
        // inside softmax before selecting the zero rows.
        Tensor valid=ne(max(scores,-1,true),scalar(-std::numeric_limits<double>::infinity(),q));
        Tensor safe=where(valid,scores,scalar(0,q));
        weights=where(valid,softmax(safe,-1),scalar(0,q));
    } else weights=scores;
    return cast(matmul(dropout(weights,dropout_p,seed,true),v),storage);
}

void validate_sdpa_stage(bool backward,const BufferPtr& output,const std::vector<BufferPtr>& inputs,const KernelParams& p) {
    require(output->backend==Backend::CUDA && output->dtype==DType::Float32,"SDPA native stages require CUDA Float32");
    require(inputs.size()>=2 && inputs.size()<=4 && inputs[0] && inputs[0]->dtype==DType::Float32,"Invalid SDPA stage inputs");
    for(size_t i=2;i<inputs.size();++i) require(!inputs[i],"Unexpected SDPA stage operand");
    int rank=p.input_rank(KernelInput::A);
    require(rank>=2 && rank<=8 && p.input_shape(KernelInput::A,rank-1)>0 && p.input_shape(KernelInput::A,rank-1)<=1024,"Unsupported SDPA stage width");
    require(output->count==inputs[0]->count*(backward?1u:2u),"SDPA stage output size mismatch");
    if(backward) require(inputs[1] && inputs[1]->dtype==DType::Float32 && inputs[1]->count==inputs[0]->count,"Invalid SDPA probability gradient");
    else if(inputs[1]) {
        require(inputs[1]->dtype==DType::Bool || inputs[1]->dtype==DType::Float32,"Invalid SDPA mask dtype");
        const int mr=p.input_rank(KernelInput::B); require(mr<=rank,"Invalid SDPA mask rank");
        for(int i=0;i<mr;++i) require(p.input_shape(KernelInput::B,i)==1 || p.input_shape(KernelInput::B,i)==p.input_shape(KernelInput::A,rank-mr+i),"Invalid SDPA mask shape");
    }
}
}
