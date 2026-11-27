#include "Core/Backend.h"
#include "Core/LocalFunction.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
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
    for(int index=0;index<4;++index){
        int base=KernelParams::input_slot(static_cast<KernelInput>(index));p.v[base]=-1;
        if(index>=int(inputs.size())){buffers.push_back(nullptr);continue;}
        const Input& input=inputs[index];
        if(count(input.shape)!=int(input.values.size()))throw std::runtime_error("Malformed native-kernel fixture");
        p.v[base]=int(input.shape.size());std::copy(input.shape.begin(),input.shape.end(),p.v+base+1);
        int stride=1;for(int axis=int(input.shape.size())-1;axis>=0;--axis){p.v[base+9+axis]=stride;stride*=input.shape[axis];}
        p.v[KernelParams::input_dtype_slot(static_cast<KernelInput>(index))]=int(input.dtype);
        BufferPtr buffer=backend_allocate(backend,input.values.size(),input.dtype);
        if(input.dtype==DType::QInt8){
            std::vector<uint8_t> codes;for(float value:input.values)codes.push_back(uint8_t(int8_t(value)));
            backend_upload_raw(buffer,codes);
        }else backend_upload(buffer,input.values);
        buffers.push_back(buffer);
    }
    std::copy(attributes.begin(),attributes.end(),p.v+64);p.v[255]=int(dtype);
    BufferPtr output=backend_allocate(backend,p.v[0],dtype);backend_dispatch(op,output,buffers,p);
    std::vector<float> result;
    if(dtype==DType::QInt8){for(unsigned char code:backend_download_raw(output))result.push_back(float(int8_t(code)));}
    else result=backend_download(output);
    visited.insert(op);
    // MPS uses Philox; numerical kernels still share the independent CPU oracle.
    const bool metal_random=backend==Backend::Metal&&(op=="uniform"||op=="normal"||op=="dropout");
    if(backend!=Backend::CPU&&!metal_random)close(op+" CPU parity",result,call(Backend::CPU,op,shape,inputs,attributes,dtype),3e-4f);
    return result;
}
void close(const std::string& label,const std::vector<float>& actual,const std::vector<float>& expected,float tolerance=2e-5f) {
    if(actual.size()!=expected.size())throw std::runtime_error(label+": native output count mismatch");
    for(size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance*(1.0f+std::fabs(expected[i])))
            throw std::runtime_error(label+": native mismatch at "+std::to_string(i)+", actual="+std::to_string(actual[i])+", expected="+std::to_string(expected[i]));
}
void flux_kernel_reference(Backend backend) {
    Input q{{1,1,1,128}, std::vector<float>(128)};
    Input table{{1,1,128}, std::vector<float>(128)};
    std::vector<float> rotated(128);
    std::iota(q.values.begin(), q.values.end(), -64.f);
    for (int d = 0; d < 128; d += 2) {
        table.values[d + 1] = 1;
        rotated[d] = -q.values[d + 1];
        rotated[d + 1] = q.values[d];
    }
    close("FLUX adjacent pairs", call(backend,"flux_rotary",q.shape,{q,table}), rotated, 0);
    Input keys{{1,1,2,128}, std::vector<float>(256,1)};
    Input values = keys;
    std::fill(values.values.begin()+128, values.values.end(), 5);
    close("FLUX joint attention", call(backend,"flux_attention",q.shape,{q,keys,values}), std::vector<float>(128,3));
    Input x{{1,2,2},{-1,1,1,-1}}, shift{{1,1,2},{1,2}}, scale{{1,1,2},{2,0}};
    const float norm = 1.f/std::sqrt(1.f+1e-6f);
    close("FLUX adaptive LayerNorm", call(backend,"flux_modulate",x.shape,{x,shift,scale}),
          {1-3*norm,2+norm,1+3*norm,2-norm});
}

void spatial_forward_parity(Backend backend) {
    Input (*const patterned)(std::vector<int>, DType, int) = [](std::vector<int> shape, DType dtype, int period) {
        Input input{shape, std::vector<float>(count(shape)), dtype};
        for (size_t i = 0; i < input.values.size(); ++i)
            input.values[i] = (int(i % period) - period / 2) / 32.0f;
        return input;
    };
    for (const elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        // Odd plane sizes and filter widths also exercise scratch and batch alignment.
        const Input input = patterned({2, 17, 17, 25}, dtype, 17);
        const Input filter = patterned({19, 17, 3, 3}, dtype, 13);
        const Input bias = patterned({19}, dtype, 7);
        call(backend, "conv2d", {2, 19, 9, 13}, {input, filter, bias}, {2, 2, 2, 2, 2, 2, 1}, dtype);
        call(backend, "conv2d", {2, 19, 9, 13},
             {patterned({2, 17, 9, 13}, dtype, 17), patterned({19, 17, 1, 1}, dtype, 13)},
             {1, 1, 0, 0, 1, 1, 1}, dtype);

        Input features{{2, 4, 5, 17}, std::vector<float>(680), dtype};
        const Input scale{{4}, {1, 2, 0.5f, -1}, dtype}, shift{{4}, {-0.25f, 0, 0.5f, -1}, dtype};
        std::vector<float> expected(features.values.size());
        for (size_t i = 0; i < features.values.size(); ++i) {
            const float sign = i % 2 ? 1.0f : -1.0f;
            const size_t channel = (i / 85) % 4;
            features.values[i] = 2.0f * float(i / 170) + sign;
            expected[i] = sign * scale.values[channel] + shift.values[channel];
        }
        close("group norm batched affine",
              call(backend, "group_norm", features.shape, {features, scale, shift}, {2, bits(1e-5f)}, dtype), expected);
    }
    // The last two elements require a separate statistics tile and affect the mean.
    Input features{{1, 1, 1, 4098}, std::vector<float>(4098, 1)};
    std::fill_n(features.values.begin(), 2048, -1.0f);
    const float mean = 2.0f / 4098.0f;
    const float inverse = 1.0f / std::sqrt(1.0f - mean * mean + 1e-5f);
    std::vector<float> expected;
    for (const float value : features.values) expected.push_back((value - mean) * inverse);
    close("group norm partial statistics tile",
          call(backend, "group_norm", features.shape, {features}, {1, bits(1e-5f)}), expected);
}
void index_limits(Backend backend) {
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16, DType::Int32}) {
        for (const std::string op : {"linear", "packed_linear"}) {
            BufferPtr input = backend_allocate(backend, 1, dtype), weight = backend_allocate(backend, 0, dtype);
            BufferPtr output = backend_allocate(backend, 0, dtype);
            KernelParams p;
            p.v[1] = op == "linear" ? 2 : 1;
            if (op == "linear") p.v[2] = 2;
            p.v[255] = int(dtype);
            for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
            for (int i = 0; i < 2; ++i) {
                int base = KernelParams::input_slot(KernelInput(i));
                p.v[base] = 2; p.v[base + 1] = i == 0 ? 2 : 0; p.v[base + 2] = 1 << 30;
                p.v[base + 9] = 1 << 30; p.v[base + 10] = 1;
                p.v[KernelParams::input_dtype_slot(KernelInput(i))] = int(dtype);
            }
            p.v[64] = 1;
            // Simulate a large storage descriptor without allocating gigabytes.
            // Empty output makes the unfixed path safe to exercise: no data reads.
            input->count = size_t(std::numeric_limits<int>::max()) + 1;
            uint64_t dispatches = backend_dispatch_count(backend);
            bool rejected = false;
            try { backend_dispatch(op, output, {input, weight}, p); }
            catch (const std::invalid_argument& error) { rejected = std::string(error.what()).find("Int32 indexing") != std::string::npos; }
            input->count = 1;
            if (!rejected || backend_dispatch_count(backend) != dispatches)
                throw std::runtime_error(op + ": oversized input reached native dispatch");
        }
    }
    BufferPtr output = backend_allocate(backend, 0, DType::Float32);
    output->count = size_t(std::numeric_limits<int>::max()) + 1;
    bool rejected = false;
    try { backend_dispatch("copy", output, {}, KernelParams{}); }
    catch (const std::invalid_argument& error) { rejected = std::string(error.what()).find("Int32 indexing") != std::string::npos; }
    output->count = 0;
    if (!rejected) throw std::runtime_error("Oversized output reached native dispatch");
}
}

void test_kernel_reference(elman::Backend backend) {
    close("gated_delta",call(backend,"gated_delta",{3},
        {{{1,2,1,3},{1,1,2,1,1,3}},{{1,2,1,2},{.5f,.5f,.5f,.5f}},{{1,1,1,1},{0}}},{1,1}),{1,1.75f,1.75f});
    index_limits(backend);
    spatial_forward_parity(backend);
    const Input quantized{{2,3},{-4,-2,0,2,4,6},DType::QInt8};
    const Input scale{{},{.5f}}, zero{{},{-2},DType::Int32};
    close("affine_quantize",call(backend,"affine_quantize",{2,3},
        {{{2,3},{-1,0,1,2,3,4}},scale,zero},{3},DType::QInt8),{-4,-2,0,2,4,6},0);
    close("affine_dequantize",call(backend,"affine_dequantize",{2,3},{quantized,scale,zero},{3}),{-1,0,1,2,3,4},0);
    close("affine_relu",call(backend,"affine_relu",{2,3},{quantized,scale,zero},{3},DType::QInt8),{-2,-2,0,2,4,6},0);
    close("affine_embedding",call(backend,"affine_embedding",{2,3},
        {quantized,scale,zero,{{2},{1,0},DType::Int32}},{3}),{2,3,4,-1,0,1},0);
    close("paged_attention",call(backend,"paged_attention",{1,2,2,2},
        {{{1,2,2,2},std::vector<float>(8,0)},{{3,2},std::vector<float>(6,0)},
         {{3,2},{0,0,2,4,6,8}},{{1,2},{2,1},DType::Int32}},{bits(1),2,0,0,1}),{4,6,4,6,4,6,4,6},0);
    close("group_norm_weight_grad", call(backend, "group_norm_weight_grad", {2},
          {{{2, 2}, {1, 2, 3, 4}}, {{2, 2}, {1, 2, 3, 4}}}, {2, bits(1e-5f)}), {0, 0}, 0);
    close("index_copy", call(backend, "index_copy", {3, 2},
          {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{3}, {2, 0, 2}, DType::Int32}, {{3, 2}, {7, 8, 9, 10, 11, 12}}},
          {0}), {9, 10, 3, 4, 11, 12}, 0);
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        close("packed_linear", call(backend, "packed_linear", {6},
              {{{2, 2}, {1, 2, 3, 4}, dtype}, {{3, 2}, {1, 0, 0, 1, 1, 1}, dtype}, {{3}, {1, 2, 3}, dtype}},
              {2, 1, 2}, dtype), {2, 4, 4, 6, 6, 10}, 0);
        Input query{{1, 2, 3, 2}, std::vector<float>(12, 0), dtype};
        Input key{{1, 1, 2, 2}, std::vector<float>(4, 0), dtype};
        Input value{{1, 1, 2, 1}, {2, 4}, dtype}, additive{{}, {0}, dtype};
        close("tiled_attention", call(backend, "tiled_attention", {1, 2, 3, 1}, {query, key, value, additive}, {bits(1), 2}, dtype),
              {3, 3, 3, 3, 3, 3}, 0);
        Input scores{{2, 4}, {2, 4, 6, 8, -2, 0, 2, 4}, dtype};
        Input mask{{2, 4}, {-1, -2, -3, -4, 1, 0, -1, -2}, dtype};
        close("attention_softmax", call(backend, "attention_softmax", {2, 4}, {scores, mask}, {1, bits(2)}, dtype),
              {.25f, .25f, .25f, .25f, .25f, .25f, .25f, .25f}, 0);
    }
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
    close("silu_mul",call(backend,"silu_mul",{3},{signed_values,two}),{-4/(1+std::exp(2.0f)),0,4/(1+std::exp(-2.0f))});
    for(elman::DType dtype:{DType::Float32,DType::Float16,DType::BFloat16})
        close("rotary_embedding",call(backend,"rotary_embedding",{2,4},{{{2,4},{1,2,3,4,5,6,7,8},dtype},{{4},{1,0,1,0},dtype},{{4},{0,1,0,1},dtype}},{},dtype),{1,-4,3,2,5,-8,7,6},0);
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
    close("scatter_source_grad",call(backend,"scatter_source_grad",{3},{{{4},{1,2,3,4}},{{3},{2,2,0},DType::Int32}},{0}),{3,3,1});
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
    std::vector<float> uniform=call(backend,"uniform",{1024},{},{42,0,bits(0),bits(1)});
    close("uniform reproducibility",call(backend,"uniform",{1024},{},{42,0,bits(0),bits(1)}),uniform);
    float average=0;for(float value:uniform){if(value<0||value>=1)throw std::runtime_error("Uniform random range");average+=value/1024;}
    if(std::fabs(average-0.5f)>0.05f)throw std::runtime_error("Uniform random mean");
    std::vector<float> normal=call(backend,"normal",{1024},{},{43,0,bits(0),bits(1)});average=0;float variance=0;
    for(float value:normal){average+=value/1024;variance+=value*value/1024;}
    if(std::fabs(average)>0.15f||std::fabs(variance-1)>0.2f)throw std::runtime_error("Normal random moments");
    // Dropout preserves its original stream; uniform has a dtype-specific grid.
    std::vector<float> masked;
    for(uint32_t i=0;i<1024;++i){
        uint32_t word=i+42u*2891336453u+1u;
        word=(word^(word>>16))*2246822519u;
        word=(word^(word>>13))*3266489917u;
        word^=word>>16;
        const float value=(float(word&16777215u)+0.5f)/16777216.f;
        masked.push_back(value>=0.25f?1/0.75f:0);
    }
    const auto dropped=call(backend,"dropout",{1024},{{{1024},std::vector<float>(1024,1)}},{42,0,bits(0.25f),1});
    if(backend==Backend::Metal){
        close("dropout reproducibility",call(backend,"dropout",{1024},{{{1024},std::vector<float>(1024,1)}},{42,0,bits(0.25f),1}),dropped,0);
        size_t kept=0;
        for(float value:dropped){
            if(value!=0&&std::abs(value-4.f/3.f)>1e-6f)throw std::runtime_error("Dropout mask scaling");
            kept+=value!=0;
        }
        if(std::abs(double(kept)/dropped.size()-.75)>.06)throw std::runtime_error("Dropout keep rate");
    }else close("dropout mask",dropped,masked);
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
    const std::vector<float> dx=call(backend,"conv2d_input_grad",x.shape,{upstream,w},conv_attributes);
    const std::vector<float> dw=call(backend,"conv2d_weight_grad",w.shape,{x,upstream},conv_attributes);
    LocalFunction objective{[&](){std::vector<float> value=call(backend,"conv2d",upstream.shape,{x,w},conv_attributes);float result=0;for(size_t i=0;i<value.size();++i)result+=value[i]*upstream.values[i];return result;}};
    for(int input=0;input<2;++input){std::vector<float>& values=input==0?x.values:w.values;const std::vector<float>& derivatives=input==0?dx:dw;
        for(size_t i=0;i<values.size();++i){float old=values[i];values[i]=old+0.001f;float positive=objective();values[i]=old-0.001f;float negative=objective();values[i]=old;close("native conv finite difference",{derivatives[i]},{(positive-negative)/0.002f},2e-4f);}
    }
    close("where",call(backend,"where",{2,2},{{{2,1},{1,0},DType::Bool},{{1,2},{2,3}},{{},{4}}}),{2,3,4,4});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Input compare_lhs{{2,1,3},{nan,-0.f,2,1,3,-2}};
    const Input compare_rhs{{1,4,1},{0,1,nan,3}};
    std::vector<float> expected_ne;
    for (int batch = 0; batch < 2; ++batch)
        for (float rhs : compare_rhs.values)
            for (int col = 0; col < 3; ++col)
                expected_ne.push_back(compare_lhs.values[batch * 3 + col] != rhs);
    close("ne interior broadcast", call(backend,"ne",{2,4,3},{compare_lhs,compare_rhs},{},DType::Bool),expected_ne,0);
    close("ne contiguous suffix", call(backend,"ne",{2,1,3},{compare_lhs,{{3},{nan,0,2}}},{},DType::Bool),{1,0,0,1,1,1},0);
    Input norm_x{{1,4,1,2},{1,2,3,4,5,6,7,8}};
    const Input norm_w{{4},{0.5f,0.6f,0.7f,0.8f}};
    const Input norm_g{{1,4,1,2},{0.2f,-0.3f,0.1f,0.4f,-0.2f,0.1f,0.3f,-0.4f}};
    const std::vector<float> norm_dx=call(backend,"group_norm_input_grad",norm_x.shape,{norm_x,norm_w,norm_g},{2,bits(1e-5f)});
    LocalFunction norm_objective{[&](){std::vector<float> y=call(backend,"group_norm",norm_x.shape,{norm_x,norm_w},{2,bits(1e-5f)});float value=0;for(size_t i=0;i<y.size();++i)value+=y[i]*norm_g.values[i];return value;}};
    for(size_t i=0;i<norm_x.values.size();++i){float old=norm_x.values[i];norm_x.values[i]=old+0.001f;float positive=norm_objective();norm_x.values[i]=old-0.001f;float negative=norm_objective();norm_x.values[i]=old;close("native group norm finite difference",{norm_dx[i]},{(positive-negative)/0.002f},2e-4f);}
    close("linear",call(backend,"linear",{2,2},{{{2,3},{1,2,3,4,5,6}},{{2,3},{1,0,-1,0,1,0}},{{2},{0.5f,-0.5f}}}),{-1.5f,1.5f,-1.5f,4.5f});
    close("argsort",call(backend,"argsort",{2,3},{{{2,3},{3,1,1,-2,0,0}}},{1,0},DType::Int32),{1,2,0,0,1,2});
    for (int length : {127, 129, 511, 513, 1025}) {
        std::vector<float> values(length), indices(length);
        for (int i = 0; i < length; ++i)
            values[i] = i % 7 == 0 ? std::numeric_limits<float>::quiet_NaN() :
                        i % 3 == 0 ? -0.f : float(i % 5);
        for (int descending : {0, 1}) {
            std::iota(indices.begin(), indices.end(), 0.f);
            std::stable_sort(indices.begin(), indices.end(), [&](int i, int j) {
                const float a = values[i], b = values[j];
                if (std::isnan(a) || std::isnan(b))
                    return descending ? std::isnan(a) && !std::isnan(b) : !std::isnan(a) && std::isnan(b);
                return descending ? a > b : a < b;
            });
            close("stable argsort ties and padding", call(backend,"argsort",{length},{{{length},values}},
                                                         {0,descending},DType::Int32), indices, 0);
        }
    }
    close("sort_backward",call(backend,"sort_backward",{2,3},{{{2,3},{1,2,3,4,5,6}},{{2,3},{1,2,0,0,1,2},DType::Int32}},{1}),{3,1,2,4,5,6});
    close("cumsum",call(backend,"cumsum",{2,3},{{{2,3},{1,2,3,-4,5,6}}},{1}),{1,3,6,-4,1,7});
    close("cumsum_backward",call(backend,"cumsum_backward",{2,3},{{{2,3},{1,2,3,-4,5,6}}},{1}),{6,5,3,7,11,6});
    close("token_presence",call(backend,"token_presence",{2,5},{{{2,4},{1,1,3,-1,0,2,4,2147483647.f},DType::Int32}},{},DType::Int32),{0,1,0,1,1,1,0,1,0,1});
    close("categorical",call(backend,"categorical",{2},{{{2,4},{.25,.25,.5,0,0,1,0,0}},{{2,4},{.25,.5,1,1,0,1,1,1}},{{2},{.5,0}}},{},DType::Int32),{2,1});
    std::vector<int> optimizer_attributes{0,0,1,0};
    for(double scalar:{.5,1.,0.,0.,.5,.75,.25,1.,1.}){
        int words[2];std::memcpy(words,&scalar,sizeof(scalar));
        optimizer_attributes.insert(optimizer_attributes.end(),words,words+2);
    }
    for(int code=16;code<=21;++code){
        const DType dtype=DType(code);
        const std::vector<Input> operands={{{4},{1,2,4,4},dtype},{{4},{1,2,4,4},dtype},{{4},{0,0,0,0}},{{4},{0,0,0,0}}};
        for(int mode=0;mode<4;++mode){
            optimizer_attributes[1]=mode;
            const std::vector<float> expected=mode==1?std::vector<float>{1,2,4,4}:mode==2?std::vector<float>{0,0,0,0}:std::vector<float>{.5f,1,2,2};
            close("low optimizer SGD",call(backend,"optimizer_step",{4},operands,optimizer_attributes,mode==0?dtype:DType::Float32),expected,0);
        }
    }
    flux_kernel_reference(backend);
    const std::vector<std::string> expected={
#define ELMAN_CPU_KERNEL(name) #name,
#define ELMAN_CPU_KERNEL4(name) #name,
#include "../Src/Kernals/cpu_dispatch.inc"
#undef ELMAN_CPU_KERNEL
#undef ELMAN_CPU_KERNEL4
    };
    for(const std::string& name:expected)if(!visited.count(name))throw std::runtime_error("Missing native kernel test: "+name);
}

int main(int argc,char** argv) {
    try{
        const Backend backend=elman::parse_backend(argc>1?argv[1]:"cpu");
        if(!elman::backend_available(backend))return 77;
        test_kernel_reference(backend);
        std::cout<<"PASS "<<visited.size()<<" native kernals on "<<elman::backend_name(backend)<<"; expected values, CPU parity, and convolution finite differences\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL native kernels: "<<error.what()<<'\n';return 1;}
}
