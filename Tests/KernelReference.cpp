#include "Core/Backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace elman;
struct Input { std::vector<int> shape; std::vector<float> values; DType dtype=DType::Float32; };
std::set<std::string> visited;
void close(const std::string& label,const std::vector<float>& actual,const std::vector<float>& expected,float tolerance);
int count(const std::vector<int>& shape) { int result=1;for(int d:shape)result*=d;return result; }
int bits(float value) { int result;std::memcpy(&result,&value,4);return result; }
std::vector<float> call(Backend backend,const std::string& op,const std::vector<int>& shape,
                        const std::vector<Input>& inputs,const std::vector<int>& attributes={},
                        DType dtype=DType::Float32) {
    KernelParams p;p.v[0]=count(shape);p.v[1]=int(shape.size());
    std::copy(shape.begin(),shape.end(),p.v+2);
    std::vector<BufferPtr> buffers;
    for(int index=0;index<3;++index){
        int base=10+17*index;p.v[base]=-1;
        if(index>=int(inputs.size())){buffers.push_back(nullptr);continue;}
        const auto& input=inputs[index];
        if(count(input.shape)!=int(input.values.size()))throw std::runtime_error("Malformed native-kernel fixture");
        p.v[base]=int(input.shape.size());std::copy(input.shape.begin(),input.shape.end(),p.v+base+1);
        int stride=1;for(int axis=int(input.shape.size())-1;axis>=0;--axis){p.v[base+9+axis]=stride;stride*=input.shape[axis];}
        p.v[252+index]=int(input.dtype);
        auto buffer=backend_allocate(backend,input.values.size(),input.dtype);backend_upload(buffer,input.values);buffers.push_back(buffer);
    }
    std::copy(attributes.begin(),attributes.end(),p.v+64);p.v[255]=int(dtype);
    auto output=backend_allocate(backend,p.v[0],dtype);backend_dispatch(op,output,buffers,p);
    auto result=backend_download(output);visited.insert(op);
    if(backend!=Backend::CPU)close(op+" CPU parity",result,call(Backend::CPU,op,shape,inputs,attributes,dtype),3e-4f);
    return result;
}
void close(const std::string& label,const std::vector<float>& actual,const std::vector<float>& expected,float tolerance=2e-5f) {
    if(actual.size()!=expected.size())throw std::runtime_error(label+": native output count mismatch");
    for(size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance*(1.0f+std::fabs(expected[i])))
            throw std::runtime_error(label+": native mismatch at "+std::to_string(i)+", actual="+std::to_string(actual[i])+", expected="+std::to_string(expected[i]));
}
}

void test_kernel_reference(elman::Backend backend) {
    const Input positive{{3},{1,2,4}};
    const Input two{{},{2}};
    close("add",call(backend,"add",{3},{positive,two}),{3,4,6});
    close("sub",call(backend,"sub",{3},{positive,two}),{-1,0,2});
    close("mul",call(backend,"mul",{3},{positive,two}),{2,4,8});
    close("div",call(backend,"div",{3},{positive,two}),{0.5f,1,2});
    close("pow",call(backend,"pow",{3},{positive,two}),{1,4,16});
    close("maximum",call(backend,"maximum",{3},{positive,two}),{2,2,4});
    close("minimum",call(backend,"minimum",{3},{positive,two}),{1,2,2});
    close("eq",call(backend,"eq",{3},{positive,two},{},DType::Bool),{0,1,0});
    close("ne",call(backend,"ne",{3},{positive,two},{},DType::Bool),{1,0,1});
    close("gt",call(backend,"gt",{3},{positive,two},{},DType::Bool),{0,0,1});
    close("ge",call(backend,"ge",{3},{positive,two},{},DType::Bool),{0,1,1});
    close("lt",call(backend,"lt",{3},{positive,two},{},DType::Bool),{1,0,0});
    close("le",call(backend,"le",{3},{positive,two},{},DType::Bool),{1,1,0});
    const Input logical{{3},{0,1,2}};
    const Input logical_other{{3},{1,1,0}};
    close("logical_and",call(backend,"logical_and",{3},{logical,logical_other},{},DType::Bool),{0,1,0});
    close("logical_or",call(backend,"logical_or",{3},{logical,logical_other},{},DType::Bool),{1,1,1});
    close("logical_not",call(backend,"logical_not",{3},{logical},{},DType::Bool),{1,0,0});
    close("sqrt",call(backend,"sqrt",{3},{positive}),{1,std::sqrt(2.0f),2});
    close("rsqrt",call(backend,"rsqrt",{3},{positive}),{1,1/std::sqrt(2.0f),0.5f});
    close("exp",call(backend,"exp",{3},{positive}),{std::exp(1.0f),std::exp(2.0f),std::exp(4.0f)});
    close("log",call(backend,"log",{3},{positive}),{0,std::log(2.0f),std::log(4.0f)});
    close("sin",call(backend,"sin",{3},{positive}),{std::sin(1.0f),std::sin(2.0f),std::sin(4.0f)});
    close("cos",call(backend,"cos",{3},{positive}),{std::cos(1.0f),std::cos(2.0f),std::cos(4.0f)});
    const Input signed_values{{3},{-2,0,2}};
    close("abs",call(backend,"abs",{3},{signed_values}),{2,0,2});
    close("sign",call(backend,"sign",{3},{signed_values}),{-1,0,1});
    close("relu",call(backend,"relu",{3},{signed_values}),{0,0,2});
    close("leaky_relu",call(backend,"leaky_relu",{3},{signed_values},{bits(0.1f)}),{-0.2f,0,2});
    for(const char* name:{"silu","swish"})close(name,call(backend,name,{3},{signed_values}),{-2/(1+std::exp(2.0f)),0,2/(1+std::exp(-2.0f))});
    std::vector<float> gelu;for(float x:signed_values.values)gelu.push_back(0.5f*x*(1+std::tanh(0.7978845608028654f*(x+0.044715f*x*x*x))));
    close("gelu",call(backend,"gelu",{3},{signed_values}),gelu);
    close("cast int",call(backend,"cast",{3},{{{3},{-2.9f,0.9f,2.9f}}},{},DType::Int32),{-2,0,2});
    const Input matrix{{2,2},{1,2,3,4}};
    close("addmm",call(backend,"addmm",{2,2},{matrix,matrix,{{2},{2,3}}},{bits(2),bits(3)}),{20,29,36,53});
    close("bmm",call(backend,"bmm",{2,1,1},{{{2,1,2},{1,2,3,4}},{{1,2,1},{5,6}}}),{17,39});
    close("matmul",call(backend,"matmul",{2,2},{matrix,matrix}),{7,10,15,22});
    close("transpose",call(backend,"transpose",{2,2},{matrix},{0,1}),{1,3,2,4});
    close("permute",call(backend,"permute",{2,2},{matrix},{1,0}),{1,3,2,4});
    for(const char* op:{"reshape","squeeze","unsqueeze","copy"})
        close(op,call(backend,op,{4},{matrix}),{1,2,3,4});
    close("split",call(backend,"split",{2,1},{matrix},{1,1,1}),{2,4});
    close("slice",call(backend,"slice",{2,1},{matrix},{1,1,1}),{2,4});
    close("gather",call(backend,"gather",{2,2},{matrix,{{2,2},{1,1,0,1},DType::Int32}},{1}),{2,2,3,4});
    close("index_select",call(backend,"index_select",{2,2},{matrix,{{2},{1,0},DType::Int32}},{0}),{3,4,1,2});
    close("repeat",call(backend,"repeat",{2,4},{matrix}),{1,2,1,2,3,4,3,4});
    close("expand",call(backend,"expand",{2,3},{{{1,3},{1,2,3}}}),{1,2,3,1,2,3});
    close("concat",call(backend,"concat",{2,3},{matrix,{{2,1},{5,6}}},{1}),{1,2,5,3,4,6});
    close("scatter duplicate",call(backend,"scatter",{4},{{{4},{1,2,3,4}},{{3},{2,2,0},DType::Int32},{{3},{5,6,7}}},{0}),{7,2,6,4});
    close("scatter_add duplicate",call(backend,"scatter_add",{4},{{{4},{1,2,3,4}},{{3},{2,2,0},DType::Int32},{{3},{5,6,7}}},{0}),{8,2,14,4});
    close("scatter_source_grad",call(backend,"scatter_source_grad",{3},{{{4},{1,2,3,4}},{{3},{2,2,0},DType::Int32}},{0}),{0,3,1});
    close("gather_grad duplicate",call(backend,"gather_grad",{2,3},{{{2,2},{1,2,3,4}},{{2,2},{2,2,0,1},DType::Int32}},{1}),{0,0,3,3,4,0});
    close("index_select_grad duplicate",call(backend,"index_select_grad",{3,2},{{{3,2},{1,2,3,4,5,6}},{{3},{2,2,0},DType::Int32}},{0}),{5,6,0,0,4,6});
    close("slice_grad step",call(backend,"slice_grad",{2,5},{{{2,2},{1,2,3,4}}},{1,1,2}),{0,1,0,2,0,0,3,0,4,0});
    close("repeat_grad",call(backend,"repeat_grad",{1,2},{{{2,4},{1,2,3,4,5,6,7,8}}}),{16,20});
    close("sum global",call(backend,"sum",{}, {matrix},{-1,0}),{10});
    for(const char* name:{"max","amax"})close(name,call(backend,name,{2},{matrix},{0,0}),{3,4});
    for(const char* name:{"min","amin"})close(name,call(backend,name,{2},{matrix},{1,0}),{1,3});
    close("mean axis",call(backend,"mean",{1,2},{matrix},{0,1}),{2,3});
    close("prod axis",call(backend,"prod",{2},{matrix},{1,0}),{2,12});
    close("prod zero gradient",call(backend,"reduce_grad",{3},{{{3},{0,2,3}},{{},{0}},{{},{2}}},{0,0,4}),{12,0,0});
    close("maximum ties gradient",call(backend,"reduce_grad",{3},{{{3},{2,2,1}},{{},{2}},{{},{4}}},{0,0,2}),{2,2,0});
    close("triu",call(backend,"triu",{2,2},{matrix},{0}),{1,2,0,4});
    close("tril",call(backend,"tril",{2,2},{matrix},{-1}),{0,0,3,0});
    close("softmax",call(backend,"softmax",{2,2},{{{2,2},{0,0,0,0}}},{1}),{0.5f,0.5f,0.5f,0.5f});
    close("log_softmax",call(backend,"log_softmax",{2,2},{{{2,2},{0,0,0,0}}},{1}),{-std::log(2.0f),-std::log(2.0f),-std::log(2.0f),-std::log(2.0f)});
    auto uniform=call(backend,"uniform",{1024},{},{42,0,bits(0),bits(1)});
    close("uniform reproducibility",call(backend,"uniform",{1024},{},{42,0,bits(0),bits(1)}),uniform);
    float average=0;for(float value:uniform){if(value<0||value>=1)throw std::runtime_error("Uniform random range");average+=value/1024;}
    if(std::fabs(average-0.5f)>0.05f)throw std::runtime_error("Uniform random mean");
    auto normal=call(backend,"normal",{1024},{},{43,0,bits(0),bits(1)});average=0;float variance=0;
    for(float value:normal){average+=value/1024;variance+=value*value/1024;}
    if(std::fabs(average)>0.15f||std::fabs(variance-1)>0.2f)throw std::runtime_error("Normal random moments");
    std::vector<float> masked;for(float value:uniform)masked.push_back(value>=0.25f?1/0.75f:0);
    close("dropout mask",call(backend,"dropout",{1024},{{{1024},std::vector<float>(1024,1)}},{42,0,bits(0.25f),1}),masked);
    const float inv=1.0f/std::sqrt(1.25f+1e-5f);
    const Input normalized{{2,4},{1,2,3,4,5,6,7,8}};
    const Input ones{{4},{1,1,1,1}};
    const Input zeros{{4},{0,0,0,0}};
    close("layer_norm",call(backend,"layer_norm",{2,4},{normalized,ones,zeros},{1,bits(1e-5f)}),{-1.5f*inv,-0.5f*inv,0.5f*inv,1.5f*inv,-1.5f*inv,-0.5f*inv,0.5f*inv,1.5f*inv});
    std::vector<float> rms;for(int row=0;row<2;++row){float square=0;for(int j=0;j<4;++j)square+=normalized.values[row*4+j]*normalized.values[row*4+j];for(int j=0;j<4;++j)rms.push_back(normalized.values[row*4+j]/std::sqrt(square/4+1e-5f));}
    close("rms_norm",call(backend,"rms_norm",{2,4},{normalized,ones},{1,bits(1e-5f)}),rms);
    close("batch_norm",call(backend,"batch_norm",{2,2,1,2},{{{2,2,1,2},{1,2,3,4,5,6,7,8}},{{2},{3,5}},{{2},{4,4}}},{bits(0.0f)}),{-1,-0.5f,-1,-0.5f,1,1.5f,1,1.5f});
    close("group_norm",call(backend,"group_norm",{1,4,1,2},{{{1,4,1,2},{1,2,3,4,5,6,7,8}},ones,zeros},{2,bits(1e-5f)}),{-1.5f*inv,-0.5f*inv,0.5f*inv,1.5f*inv,-1.5f*inv,-0.5f*inv,0.5f*inv,1.5f*inv});
    close("mse_loss",call(backend,"mse_loss",{}, {matrix,{{2,2},{0,1,2,3}}}),{1});
    close("cross_entropy",call(backend,"cross_entropy",{}, {{{2,3},{0,0,0,0,0,0}},{{2},{0,2},DType::Int32}}),{std::log(3.0f)});
    close("cross_entropy_grad",call(backend,"cross_entropy_grad",{2,3},{{{2,3},{0,0,0,0,0,0}},{{2},{0,2},DType::Int32},{{},{2}}}),{-2.0f/3,1.0f/3,1.0f/3,1.0f/3,1.0f/3,-2.0f/3});
    Input convolution{{1,2,3,3},{1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9}};
    Input weights{{2,1,2,2},{1,1,1,1,2,2,2,2}};
    close("grouped conv2d",call(backend,"conv2d",{1,2,2,2},{convolution,weights},{1,1,0,0,1,1,2}),{12,16,24,28,24,32,48,56});
    close("max_pool2d",call(backend,"max_pool2d",{1,2,2,2},{convolution},{2,2,1,1,0,0,1}),{5,6,8,9,5,6,8,9});
    close("avg_pool2d",call(backend,"avg_pool2d",{1,2,2,2},{convolution},{2,2,1,1,0,0,1}),{3,4,6,7,3,4,6,7});
    const Input pool_input{{1,1,3,3},{1,2,3,4,5,6,7,8,9}};
    const Input pool_grad{{1,1,2,2},{1,1,1,1}};
    close("max_pool2d_grad",call(backend,"max_pool2d_grad",{1,1,3,3},{pool_input,pool_grad},{2,2,1,1,0,0,1}),{0,0,0,0,1,1,0,1,1});
    close("avg_pool2d_grad",call(backend,"avg_pool2d_grad",{1,1,3,3},{pool_input,pool_grad},{2,2,1,1,0,0,1}),{0.25f,0.5f,0.25f,0.5f,1,0.5f,0.25f,0.5f,0.25f});
    close("slice_backward",call(backend,"slice_backward",{2,5},{{{2,2},{1,2,3,4}}},{1,1,2}),{0,1,0,2,0,0,3,0,4,0});
    close("gather_backward",call(backend,"gather_backward",{2,3},{{{2,2},{1,2,3,4}},{{2,2},{2,2,0,1},DType::Int32}},{1}),{0,0,3,3,4,0});
    close("index_select_backward",call(backend,"index_select_backward",{3,2},{{{3,2},{1,2,3,4,5,6}},{{3},{2,2,0},DType::Int32}},{0}),{5,6,0,0,4,6});
    close("repeat_backward",call(backend,"repeat_backward",{1,2},{{{2,4},{1,2,3,4,5,6,7,8}}}),{16,20});
    // Finite differences test native grouped, padded, strided, dilated convolution derivatives.
    Input x{{1,2,4,4},std::vector<float>(32)};
    Input w{{2,1,2,2},std::vector<float>(8)};
    for(size_t i=0;i<x.values.size();++i)x.values[i]=0.02f*float(int(i)%9-4);
    for(size_t i=0;i<w.values.size();++i)w.values[i]=0.03f*float(int(i)-3);
    const std::vector<int> conv_attributes{2,2,1,1,2,2,2};
    Input upstream{{1,2,2,2},{0.2f,-0.3f,0.1f,0.4f,-0.2f,0.1f,0.3f,-0.4f}};
    const auto dx=call(backend,"conv2d_input_grad",x.shape,{upstream,w},conv_attributes);
    const auto dw=call(backend,"conv2d_weight_grad",w.shape,{x,upstream},conv_attributes);
    auto objective=[&](){auto value=call(backend,"conv2d",upstream.shape,{x,w},conv_attributes);float result=0;for(size_t i=0;i<value.size();++i)result+=value[i]*upstream.values[i];return result;};
    for(int input=0;input<2;++input){auto& values=input==0?x.values:w.values;const auto& derivatives=input==0?dx:dw;
        for(size_t i=0;i<values.size();++i){float old=values[i];values[i]=old+0.001f;float positive=objective();values[i]=old-0.001f;float negative=objective();values[i]=old;close("native conv finite difference",{derivatives[i]},{(positive-negative)/0.002f},2e-4f);}
    }
    const std::vector<std::string> expected={
#define ELMAN_CPU_KERNEL(name) #name,
#include "../Src/Operators/cpu_dispatch.inc"
#undef ELMAN_CPU_KERNEL
    };
    for(const auto& name:expected)if(!visited.count(name))throw std::runtime_error("Missing native kernel test: "+name);
}

int main(int argc,char** argv) {
    try{
        const auto backend=elman::parse_backend(argc>1?argv[1]:"cpu");
        if(!elman::backend_available(backend))throw std::runtime_error("Requested native test backend unavailable");
        test_kernel_reference(backend);
        std::cout<<"PASS "<<visited.size()<<" native operators on "<<elman::backend_name(backend)<<"; expected values, CPU parity, and convolution finite differences\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL native kernels: "<<error.what()<<'\n';return 1;}
}
