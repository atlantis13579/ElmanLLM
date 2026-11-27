#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include "Core/AllocationPool.h"
#include "Core/LocalFunction.h"
#include "Core/Execution.h"
#include "Core/Graph.h"
#include "Core/TensorDispatch.h"
#include "Inference/LLM/Sampling.h"
#include "Optimizer/Optimizer.h"

namespace {

namespace attention {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double attention operation was accepted");
}
void near(const std::vector<double>& values, const std::vector<double>& expected) {
    check(values.size() == expected.size(), "Double attention result size differs");
    for (size_t i = 0; i < values.size(); ++i)
        check(std::abs(values[i] - expected[i]) < 2e-14, "Double attention result differs");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
int bits(float value) { int result; std::memcpy(&result,&value,sizeof(value)); return result; }
void raw_operations(Backend backend) {
    Tensor scores=Tensor::zeros({2,3},backend,false,DType::Float64);
    Tensor mask=Tensor::zeros({3},backend,false,DType::Float64);
    Tensor query=Tensor::zeros({1,2,2,2},backend,false,DType::Float64);
    Tensor key=Tensor::from_double_data({1,1,2,2},{1,0,0,1},backend);
    Tensor value=Tensor::from_double_data({1,1,2,1},{2,4},backend);
    Tensor bias=Tensor::zeros({2},backend,false,DType::Float64);
    struct Case { const char* name; Shape shape; std::vector<Tensor> inputs; std::vector<int> attrs; std::vector<double> expected; };
    const std::vector<Case> cases{
        {"attention_softmax",{2,3},{scores,mask},{1,bits(1)},std::vector<double>(6,1./3.)},
        {"tiled_attention",{1,2,2,1},{query,key,value,bias},{bits(1),1,0,0,0},std::vector<double>(4,3.)},
        {"tiled_attention",{1,2,2,1},{query,key,value,bias},{bits(1),1,0,0,1},std::vector<double>(4,3.)}
    };
    for (const Case& test:cases) {
        const bool tiled=std::string(test.name)=="tiled_attention";
        Tensor output=Tensor::zeros(test.shape,backend,false,DType::Float64);
        KernelParams valid=descriptor(output,test.inputs);
        for (size_t i=0;i<test.attrs.size();++i) valid.v[KernelParams::kAttributeSlot+i]=test.attrs[i];
        std::vector<BufferPtr> inputs;
        for (const elman::Tensor& input:test.inputs) inputs.push_back(input.node()->value);
        backend_dispatch(test.name,output.node()->value,inputs,valid);
        near(output.double_data(),test.expected);
        const std::vector<uint8_t> before=output.raw_data();
        std::vector<std::pair<int,int>> changes{
            {KernelParams::kOutputCountSlot,0},{KernelParams::kOutputRankSlot,-1},{KernelParams::kOutputRankSlot,9},
            {KernelParams::kOutputShapeSlot,7},{KernelParams::kOutputDTypeSlot,int(DType::Float32)},
            {KernelParams::kAttributeSlot+7,2}};
        for (size_t i=0;i<inputs.size();++i) {
            const KernelInput input=KernelInput(i);
            const int slot=KernelParams::input_slot(input);
            changes.insert(changes.end(),{{slot,-1},{slot,9},{slot+KernelParams::kInputShapeOffset,-1},
                {slot+KernelParams::kInputShapeOffset,7},{slot+KernelParams::kInputStrideOffset,0},
                {KernelParams::input_dtype_slot(input),int(DType::Float32)}});
        }
        const std::vector<std::pair<int, int>> attributes=tiled ? std::vector<std::pair<int,int>>{{1,0},{1,-1},{2,2},{4,2}} :
                                      std::vector<std::pair<int,int>>{{0,-1},{0,2},{2,1},{3,1},{4,1}};
        for (std::pair<int, int> change:attributes) changes.emplace_back(KernelParams::kAttributeSlot+change.first,change.second);
        if (!tiled) changes.emplace_back(KernelParams::input_slot(KernelInput::C),0);
        for (std::pair<int, int> change:changes) {
            KernelParams p=valid;p.v[change.first]=change.second;
            rejects([&] { backend_dispatch(test.name,output.node()->value,inputs,p); });
            check(output.raw_data()==before,"Rejected Double attention descriptor changed output");
        }
        rejects([&] { backend_dispatch(test.name,output.node()->value,{},valid); });
        std::vector<BufferPtr> extra=inputs;extra.push_back(inputs[0]);
        rejects([&] { backend_dispatch(test.name,output.node()->value,extra,valid); });
        for (size_t i=0;i<inputs.size();++i) {
            std::vector<BufferPtr> missing=inputs;missing[i].reset();
            rejects([&] { backend_dispatch(test.name,output.node()->value,missing,valid); });
            std::vector<Tensor> wrong=test.inputs;wrong[i]=cast(wrong[i],DType::Float32);
            rejects([&] { dispatch_tensor_into(test.name,output,wrong,test.attrs); });
            Tensor root=Tensor::zeros({int(test.inputs[i].numel()+output.numel()+1)},backend,false,DType::Float64);
            std::vector<Tensor> alias=test.inputs;alias[i]=contiguous_view(root,test.inputs[i].shape());
            alias[i].assign_double(test.inputs[i].double_data());
            rejects([&] { dispatch_tensor_into(test.name,contiguous_view(root,test.shape,1),alias,test.attrs); });
            if (tiled || i!=0) rejects([&] { dispatch_tensor_into(test.name,contiguous_view(root,test.shape),alias,test.attrs); });
            else {
                dispatch_tensor_into(test.name,contiguous_view(root,test.shape),alias,test.attrs);
                near(contiguous_view(root,test.shape).double_data(),test.expected);
                alias[i].assign_double(test.inputs[i].double_data());
            }
            Tensor disjoint=contiguous_view(root,test.shape,test.inputs[i].numel());
            dispatch_tensor_into(test.name,disjoint,alias,test.attrs);
            near(disjoint.double_data(),test.expected);
            if (backend==Backend::CUDA) {
                wrong=test.inputs;wrong[i]=wrong[i].to(Backend::CPU);
                rejects([&] { dispatch_tensor_into(test.name,output,wrong,test.attrs); });
            }
        }
    }
    rejects([&] { tiled_attention(query,key,value,bias,1,0); });
    rejects([&] { tiled_attention(query,key,value,bias,1,1,false,0,AttentionTiling(2)); });
    Tensor empty_key=Tensor::zeros({1,1,0,2},backend,false,DType::Float64);
    Tensor empty_value=Tensor::zeros({1,1,0,1},backend,false,DType::Float64);
    Tensor scalar=Tensor::zeros({},backend,false,DType::Float64);
    Tensor output=Tensor::ones({1,2,2,1},backend,false,DType::Float64);
    dispatch_tensor_into("tiled_attention",output,{query,empty_key,empty_value,scalar},{bits(1),1,0,0,0});
    near(output.double_data(),std::vector<double>(4,0.));
    Tensor empty_query=Tensor::zeros({1,2,0,2},backend,true,DType::Float64);
    Tensor empty=tiled_attention(empty_query,key,value,scalar);
    empty.backward(Tensor::zeros(empty.shape(),backend,false,DType::Float64));
    check(empty.numel()==0 && empty_query.double_grad().empty(),"Empty Double attention gradient differs");
}
void workspace(Backend backend) {
    AllocationPool pool(backend,0,false);
    Tensor q=Tensor::zeros({1,4,17,3},backend,false,DType::Float64);
    Tensor k=Tensor::zeros({1,2,19,3},backend,false,DType::Float64);
    Tensor v=Tensor::ones({1,2,19,5},backend,false,DType::Float64);
    Tensor m=Tensor::zeros({},backend,false,DType::Float64);
    Tensor output=Tensor::zeros({1,4,17,5},backend,false,DType::Float64);
    for (int mode:{0,1}) {
        const size_t baseline=memory_stats(backend).live_bytes();
        const size_t required=(mode ? 17u*19u : 4u*3u*19u)*sizeof(double);
        const std::vector<int> attrs{bits(1),3,0,0,mode};
        {
            MemoryLimit limit(backend,baseline+required);
            dispatch_tensor_into("tiled_attention",output,{q,k,v,m},attrs);
            check(limit.peak_bytes()==baseline+required,"Double attention workspace differs from its tile bound");
        }
        near(output.double_data(),std::vector<double>(output.numel(),1.));
        bool rejected=false;
        {
            MemoryLimit limit(backend,baseline+required-1);
            try { dispatch_tensor_into("tiled_attention",output,{q,k,v,m},attrs); }
            catch (const MemoryLimitExceeded&) { rejected=true; }
        }
        check(rejected,"Double attention bypassed its workspace budget");
        check(memory_stats(backend).live_bytes()==baseline,"Double attention leaked workspace");
    }
}
void execution(Backend backend) {
    for (bool pooled:{false,true}) for (bool asynchronous:{false,true}) {
        AllocationPool pool(backend,0,pooled);
        ExecutionStream producer(backend),consumer(backend);
        for (int mode:{0,1}) for (int iteration=0;iteration<2;++iteration) {
            Tensor query,key,value,mask,output;
            {
                StreamScope scope(producer,asynchronous,true);
                query=Tensor::from_double_data({10},{77,0,0,0,0,0,0,0,0,88},backend,true);
                key=Tensor::from_double_data({6},{33,1,0,0,1,44},backend,true);
                value=Tensor::from_double_data({4},{55,2,4,66},backend,true);
                mask=Tensor::from_double_data({3},{99,0,11},backend,true);
                output=tiled_attention(contiguous_view(query,{1,2,2,2},1),contiguous_view(key,{1,1,2,2},1),
                    contiguous_view(value,{1,1,2,1},1),contiguous_view(mask,{},1),2,1,false,0,AttentionTiling(mode));
            }
            {
                StreamScope scope(consumer,asynchronous,true);
                near(output.double_data(),{3,3,3,3});
                output.backward(Tensor::ones(output.shape(),backend,false,DType::Float64));
                near(query.double_grad(),{0,-.25,.25,-.25,.25,-.25,.25,-.25,.25,0});
                near(key.double_grad(),std::vector<double>(6,0.));
                near(value.double_grad(),{0,2,2,0}); near(mask.double_grad(),{0,0,0});
                check(query.double_data()==std::vector<double>{77,0,0,0,0,0,0,0,0,88},"Double attention mutated query storage");
                Tensor scores=Tensor::zeros({2,3},backend,true,DType::Float64);
                Tensor bias=Tensor::zeros({3},backend,true,DType::Float64);
                Tensor p=attention_softmax(scores,bias,2);
                p.backward(Tensor::from_double_data({2,3},{1,2,3,4,5,6},backend));
                near(scores.double_grad(),{-1./6.,0,1./6.,-1./6.,0,1./6.});
                near(bias.double_grad(),{-2./3.,0,2./3.});
            }
        }
        producer.synchronize();consumer.synchronize();
        if (pooled) check(pool.stats().reuses>0,"Double attention pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before=memory_stats(backend).live_bytes();
        raw_operations(backend);workspace(backend);execution(backend);
        check(memory_stats(backend).live_bytes()==before,"Double attention leaked storage");
        std::cout << "Float64 attention: " << checks << " passed, 0 failed\n";
}

}  // namespace attention

namespace compatibility {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid compatibility operation was accepted");
}
void scalar_storage(Backend backend) {
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        for (elman::DType dtype : all_dtypes()) {
            std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, 1));
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(0xa5 ^ (i * 37));
            const unsigned remaining = dtype_info(dtype).bits % 8;
            if (remaining) bytes.back() &= uint8_t((1u << remaining) - 1);
            Tensor input;
            {
                StreamScope scope(producer, asynchronous, true);
                input = Tensor::from_raw({}, dtype, bytes, backend);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                for (int a : {-1, 0}) for (int b : {-1, 0}) {
                    Tensor value = transpose(input, a, b);
                    check(value.shape().empty() && value.dtype() == dtype && value.backend() == backend, "Scalar transpose descriptor differs");
                    check(value.node()->value == input.node()->value, "Scalar transpose lost shared storage");
                    check(value.raw_data() == bytes, "Scalar transpose lost storage bits");
                }
                for (int bad : {-2, 1, INT32_MIN, INT32_MAX}) {
                    rejects([&] { transpose(input, bad, 0); });
                    rejects([&] { transpose(input, 0, bad); });
                }
            }
        }
        producer.synchronize(); consumer.synchronize();
    }
    rejects([&] { transpose(Tensor(), 0, 0); });
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16, DType::Float64}) {
        Tensor parent = dtype == DType::Float64 ? Tensor::from_double_data({3}, {1., 1. + 0x1p-40, 3.}, backend, true) :
            Tensor::from_data({3}, {1, 2, 3}, backend, true, dtype);
        Tensor value = transpose(contiguous_view(parent, {}, 1), -1, 0);
        const double derivative = dtype == DType::Float64 ? 3.25 + 0x1p-40 : 3.25;
        value.backward(dtype == DType::Float64 ? Tensor::from_double_data({}, {derivative}, backend) :
            Tensor::from_data({}, {float(derivative)}, backend, false, dtype));
        check(value.node()->value->raw_pointer() == contiguous_view(parent, {}, 1).node()->value->raw_pointer(), "Scalar transpose lost its view offset");
        if (dtype == DType::Float64) check(parent.double_grad() == std::vector<double>{0., derivative, 0.}, "Double scalar transpose derivative differs");
        else check(parent.grad() == std::vector<float>{0, 3.25f, 0}, "Scalar transpose derivative differs");
    }
}
void raw_contracts(Backend backend) {
    Tensor scalar = Tensor::from_double_data({}, {1. + 0x1p-40}, backend);
    for (int a : {-1, 0}) for (int b : {-1, 0})
        check(dispatch_tensor("transpose", {}, {scalar}, {a, b}, DType::Float64).raw_data() == scalar.raw_data(), "Raw scalar transpose lost precision");
    for (int bad : {-2, 1, INT32_MIN, INT32_MAX}) {
        rejects([&] { dispatch_tensor("transpose", {}, {scalar}, {bad, 0}, DType::Float64); });
        rejects([&] { dispatch_tensor("transpose", {}, {scalar}, {0, bad}, DType::Float64); });
    }
    Tensor source = Tensor::from_double_data({2, 2}, {1, 2, 3, 4}, backend);
    Tensor narrow = cast(source, DType::Float32);
    for (const char* op : {"swish", "repeat_backward", "slice_backward"}) {
        const std::vector<int> attrs = std::string(op) == "slice_backward" ? std::vector<int>{1, 0, 1} : std::vector<int>{};
        Tensor wrong = std::string(op) == "swish" ? cast(source, DType::Int32) : narrow;
        rejects([&] { dispatch_tensor(op, {2, 2}, {wrong}, attrs, DType::Float64); });
        rejects([&] { dispatch_tensor(op, {2, 2}, {source, source}, attrs, DType::Float64); });
        if (backend == Backend::CUDA)
            rejects([&] { dispatch_tensor_into(op, Tensor::zeros({2, 2}, backend, false, DType::Float64), {source.to(Backend::CPU)}, attrs); });
    }
    rejects([&] { dispatch_tensor("repeat_backward", {2, 3}, {source}, {}, DType::Float64); });
    rejects([&] { dispatch_tensor("slice_backward", {2, 2}, {source}, {1, 0, 0}, DType::Float64); });
    rejects([&] { dispatch_tensor("slice_backward", {2, 2}, {source}, {1, 1, 1}, DType::Float64); });
    rejects([&] { dispatch_tensor_into("repeat_backward", source, {source}); });
    rejects([&] { dispatch_tensor_into("slice_backward", source, {source}, {1, 0, 1}); });
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        scalar_storage(backend); raw_contracts(backend);
        check(memory_stats(backend).live_bytes() == before, "Compatibility operations leaked storage");
        std::cout << "Float64 compatibility: " << checks << " passed, 0 failed\n";
}

}  // namespace compatibility

namespace fused {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double fused operation was accepted");
}
void near(const std::vector<double>& values, const std::vector<double>& expected) {
    check(values.size() == expected.size(), "Double fused result size differs");
    for (size_t i = 0; i < values.size(); ++i)
        check(std::abs(values[i] - expected[i]) < 2e-14, "Double fused result differs");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void raw_operations(Backend backend) {
    Tensor x = Tensor::from_double_data({2,4},{1,2,3,4,5,6,7,8},backend);
    Tensor zero = Tensor::zeros({2,4},backend,false,DType::Float64);
    Tensor one = Tensor::ones({2,4},backend,false,DType::Float64);
    struct Case { const char* name; std::vector<Tensor> inputs; std::vector<double> expected; };
    const std::vector<Case> cases{
        {"silu_mul",{zero,x},std::vector<double>(8,0.)},
        {"rotary_embedding",{x,zero,one},{-3,-4,1,2,-7,-8,5,6}}
    };
    for (const Case& test : cases) {
        Tensor output = Tensor::zeros(x.shape(),backend,false,DType::Float64);
        const KernelParams valid = descriptor(output,test.inputs);
        std::vector<BufferPtr> inputs;
        for (const elman::Tensor& input : test.inputs) inputs.push_back(input.node()->value);
        backend_dispatch(test.name,output.node()->value,inputs,valid);
        near(output.double_data(),test.expected);
        const std::vector<uint8_t> before = output.raw_data();
        const int a = KernelParams::input_slot(KernelInput::A);
        for (std::pair<int, int> change : std::vector<std::pair<int,int>>{
                {a,-1},{a,9},{a,1},{a+KernelParams::kInputShapeOffset,-1},{a+KernelParams::kInputShapeOffset,3},
                {a+KernelParams::kInputStrideOffset,1},{KernelParams::kOutputCountSlot,0},
                {KernelParams::kOutputRankSlot,-1},{KernelParams::kOutputRankSlot,9},{KernelParams::kOutputShapeSlot,3},
                {KernelParams::kInputDTypeSlot,int(DType::Float32)},{KernelParams::kOutputDTypeSlot,int(DType::Float32)},
                {KernelParams::input_slot(KernelInput::D),0}}) {
            KernelParams p = valid; p.v[change.first] = change.second;
            rejects([&] { backend_dispatch(test.name,output.node()->value,inputs,p); });
            check(output.raw_data() == before,"Rejected Double fused descriptor changed output");
        }
        rejects([&] { backend_dispatch(test.name,output.node()->value,{},valid); });
        std::vector<BufferPtr> extra = inputs; extra.push_back(x.node()->value);
        rejects([&] { backend_dispatch(test.name,output.node()->value,extra,valid); });
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<BufferPtr> missing = inputs; missing[i].reset();
            rejects([&] { backend_dispatch(test.name,output.node()->value,missing,valid); });
            std::vector<Tensor> wrong = test.inputs; wrong[i] = cast(wrong[i],DType::Float32);
            rejects([&] { dispatch_tensor_into(test.name,output,wrong); });
            Tensor shared = Tensor::zeros({17},backend,false,DType::Float64);
            std::vector<Tensor> alias = test.inputs; alias[i] = contiguous_view(shared,x.shape());
            alias[i].assign_double(test.inputs[i].double_data());
            rejects([&] { dispatch_tensor_into(test.name,contiguous_view(shared,x.shape(),1),alias); });
            if (std::string(test.name)=="rotary_embedding")
                rejects([&] { dispatch_tensor_into(test.name,contiguous_view(shared,x.shape()),alias); });
            else {
                dispatch_tensor_into(test.name,contiguous_view(shared,x.shape()),alias);
                near(contiguous_view(shared,x.shape()).double_data(),test.expected);
                alias[i].assign_double(test.inputs[i].double_data());
            }
            Tensor disjoint = contiguous_view(shared,x.shape(),8);
            dispatch_tensor_into(test.name,disjoint,alias);
            near(disjoint.double_data(),test.expected);
        }
        if (backend==Backend::CUDA) {
            std::vector<Tensor> wrong=test.inputs; wrong[0]=wrong[0].to(Backend::CPU);
            rejects([&] { dispatch_tensor_into(test.name,output,wrong); });
        }
    }
    Tensor odd = Tensor::zeros({3},backend,false,DType::Float64);
    Tensor scalar = Tensor::ones({},backend,false,DType::Float64);
    rejects([&] { rotary_embedding(odd,scalar,scalar); });
    rejects([&] { dispatch_tensor("rotary_embedding",odd.shape(),{odd,scalar,scalar},{},DType::Float64); });
    rejects([&] { rotary_embedding(scalar,scalar,scalar); });
    rejects([&] { dispatch_tensor("rotary_embedding",{2,4},{reshape(x,{1,8}),one,one},{},DType::Float64); });
    Tensor small=Tensor::zeros({1,4},backend,false,DType::Float64);
    rejects([&] { dispatch_tensor("rotary_embedding",x.shape(),{small,one,one},{},DType::Float64); });
    Tensor input=Tensor::from_double_data({3},{-1,0,1},backend,true);
    const double slope=1.+std::ldexp(1.,-40);
    Tensor activation=leaky_relu(input,slope);
    check(activation.double_data()==std::vector<double>{-slope,0,1},"Double leaky ReLU rounded its slope");
    activation.backward(Tensor::ones({3},backend,false,DType::Float64));
    check(input.double_grad()==std::vector<double>{slope,slope,1},"Double leaky ReLU gradient rounded its slope");
    Tensor output=Tensor::zeros({3},backend,false,DType::Float64);
    rejects([&] { dispatch_tensor_into("leaky_relu",output,{input},{0,0,2}); });
    Tensor empty=Tensor::zeros({2,0},backend,true,DType::Float64);
    Tensor value=rotary_embedding(empty,scalar,scalar);
    value.backward(Tensor::zeros(value.shape(),backend,false,DType::Float64));
    check(value.numel()==0 && empty.double_grad().empty(),"Empty Double rotary result differs");
}
void execution(Backend backend) {
    for (bool pooled : {false,true}) for (bool asynchronous : {false,true}) {
        AllocationPool pool(backend,0,pooled);
        ExecutionStream producer(backend),consumer(backend);
        for (int iteration=0;iteration<3;++iteration) {
            Tensor input,cosine,sine,output;
            {
                StreamScope scope(producer,asynchronous,true);
                input=Tensor::from_double_data({6},{99,1,2,3,4,66},backend,true);
                cosine=Tensor::from_double_data({3},{77,0,88},backend,true);
                sine=Tensor::from_double_data({3},{33,1,44},backend,true);
                output=rotary_embedding(contiguous_view(input,{4},1),contiguous_view(cosine,{},1),contiguous_view(sine,{},1));
            }
            {
                StreamScope scope(consumer,asynchronous,true);
                near(output.double_data(),{-3,-4,1,2});
                output.backward(Tensor::from_double_data({4},{1,2,3,4},backend));
                near(input.double_grad(),{0,3,4,-1,-2,0});
                near(cosine.double_grad(),{0,30,0}); near(sine.double_grad(),{0,0,0});
                check(input.double_data()==std::vector<double>{99,1,2,3,4,66},"Double rotary mutated its input");
                Tensor gate=Tensor::from_double_data({3},{7,0,9},backend,true);
                Tensor up=Tensor::from_double_data({2},{2,4},backend,true);
                Tensor value=silu_mul(contiguous_view(gate,{},1),up);
                near(value.double_data(),{0,0});
                value.backward(Tensor::from_double_data({2},{3,5},backend));
                near(gate.double_grad(),{0,13,0}); near(up.double_grad(),{0,0});
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses>0,"Double fused pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before=memory_stats(backend).live_bytes();
        raw_operations(backend); execution(backend);
        check(memory_stats(backend).live_bytes()==before,"Double fused operation leaked storage");
        std::cout << "Float64 fused: " << checks << " passed, 0 failed\n";
}

}  // namespace fused

namespace indexing {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double indexing operation was accepted");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs, int axis) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    p.v[KernelParams::kAttributeSlot] = axis;
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void raw_operations(Backend backend) {
    Tensor values = Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend);
    Tensor ids = Tensor::from_int_data({2, 3}, {1, 1, 0, 0, 2, 2}, backend);
    Tensor vector = Tensor::from_int_data({3}, {2, 2, 0}, backend);
    Tensor permutation = Tensor::from_int_data({2, 3}, {2, 0, 1, 1, 2, 0}, backend);
    Tensor base = Tensor::from_double_data({2, 3}, std::vector<double>(6, 10.), backend);
    struct Case { const char* name; std::vector<Tensor> inputs; std::vector<double> expected; };
    std::vector<Case> cases{
        {"gather_grad", {values, ids}, {3, 3, 0, 4, 0, 11}},
        {"gather_backward", {values, ids}, {3, 3, 0, 4, 0, 11}},
        {"index_select_grad", {values, vector}, {3, 0, 3, 6, 0, 9}},
        {"index_select_backward", {values, vector}, {3, 0, 3, 6, 0, 9}},
        {"scatter", {base, ids, values}, {3, 2, 10, 4, 10, 6}},
        {"scatter_add", {base, ids, values}, {13, 13, 10, 14, 10, 21}},
        {"scatter_source_grad", {values, ids}, {2, 2, 1, 4, 6, 6}},
        {"sort_backward", {values, permutation}, {2, 3, 1, 6, 4, 5}},
        {"repeat_grad", {Tensor::ones({4, 6}, backend, false, DType::Float64)}, std::vector<double>(6, 4.)}
    };
    for (const Case& test : cases) {
        Tensor output = Tensor::zeros({2, 3}, backend, false, DType::Float64);
        KernelParams valid = descriptor(output, test.inputs, 1);
        std::vector<BufferPtr> inputs;
        for (const elman::Tensor& input : test.inputs) inputs.push_back(input.node()->value);
        backend_dispatch(test.name, output.node()->value, inputs, valid);
        check(output.double_data() == test.expected, "Raw Double indexing result differs");
        const std::vector<uint8_t> before = output.raw_data();
        const int slot = KernelParams::input_slot(KernelInput::A);
        for (std::pair<int, int> change : std::vector<std::pair<int, int>>{
            {slot, -1}, {slot, 9}, {slot + KernelParams::kInputShapeOffset, -1},
            {slot + KernelParams::kInputShapeOffset, 3}, {slot + KernelParams::kInputStrideOffset, 1},
            {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, -1}, {KernelParams::kOutputRankSlot, 9},
            {KernelParams::kOutputShapeSlot, 3}, {KernelParams::kInputDTypeSlot, int(DType::Float32)},
            {KernelParams::kOutputDTypeSlot, int(DType::Float32)}}) {
            KernelParams p = valid; p.v[change.first] = change.second;
            rejects([&] { backend_dispatch(test.name, output.node()->value, inputs, p); });
            check(output.raw_data() == before, "Rejected Double indexing descriptor changed output");
        }
        rejects([&] { backend_dispatch(test.name, output.node()->value, {}, valid); });
        std::vector<BufferPtr> extra = inputs; extra.push_back(values.node()->value);
        rejects([&] { backend_dispatch(test.name, output.node()->value, extra, valid); });
        if (test.inputs.size() > 1) {
            KernelParams p = valid; p.v[KernelParams::kAttributeSlot] = 2;
            rejects([&] { backend_dispatch(test.name, output.node()->value, inputs, p); });
            std::vector<Tensor> invalid = test.inputs;
            invalid[1] = Tensor::from_int_data(test.inputs[1].shape(), std::vector<int32_t>(test.inputs[1].numel(), -1), backend);
            rejects([&] { dispatch_tensor(test.name, {2, 3}, invalid, {1}, DType::Float64); });
        }
    }
    rejects([&] { dispatch_tensor("sort_backward", {2, 3}, {values, ids}, {1}, DType::Float64); });
    for (const char* op : {"scatter", "scatter_add"}) {
        Tensor destination = Tensor::from_double_data({2, 3}, std::vector<double>(6, 10.), backend);
        std::vector<double> expected = dispatch_tensor(op, {2, 3}, {destination, ids, values}, {1}, DType::Float64).double_data();
        dispatch_tensor_into(op, destination, {destination, ids, values}, {1});
        check(destination.double_data() == expected, "In-place Double scatter differs");
        rejects([&] { dispatch_tensor_into(op, destination, {destination, ids, destination}, {1}); });
    }
    Tensor root = Tensor::zeros({7}, backend, false, DType::Float64);
    rejects([&] { dispatch_tensor_into("gather_grad", contiguous_view(root, {2, 3}, 1), {contiguous_view(root, {2, 3}), ids}, {1}); });
    Tensor empty = Tensor::zeros({2, 0}, backend, false, DType::Float64);
    Tensor no_indices = Tensor::from_int_data({2, 0}, {}, backend);
    check(dispatch_tensor("gather_grad", {2, 3}, {empty, no_indices}, {1}, DType::Float64).double_data() == std::vector<double>(6, 0.),
          "Empty Double gather derivative differs");
    Tensor output = Tensor::zeros({2, 3}, backend, false, DType::Float64);
    const std::vector<uint8_t> before = output.raw_data();
    bool limited = false;
    {
        MemoryLimit limit(Backend::CPU, memory_stats(Backend::CPU).live_bytes());
        try { dispatch_tensor_into("gather_grad", output, {values, ids}, {1}); }
        catch (const MemoryLimitExceeded&) { limited = true; }
    }
    check(limited && output.raw_data() == before, "Double indexing did not enforce its host workspace limit");
    if (backend == Backend::CUDA) rejects([&] { dispatch_tensor("gather_grad", {2, 3}, {values, ids.to(Backend::CPU)}, {1}, DType::Float64); });
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 3; ++iteration) {
            Tensor input, output;
            {
                StreamScope scope(producer, asynchronous, true);
                input = Tensor::from_double_data({5}, {123, 3, 1, 2, 456}, backend, true);
                Tensor selected = gather(contiguous_view(input, {3}, 1), 0, Tensor::from_int_data({4}, {2, 1, 2, 0}, backend));
                std::pair<Tensor, Tensor> ordered = sort(selected, 0);
                check(ordered.second.int_data() == std::vector<int32_t>{1, 0, 2, 3}, "Double stable sort indices differ");
                output = repeat(ordered.first, {2});
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                check(output.double_data() == std::vector<double>{1, 2, 2, 3, 1, 2, 2, 3}, "Double indexing stream result differs");
                output.backward(Tensor::from_double_data({8}, {1, 2, 3, 4, 5, 6, 7, 8}, backend));
                check(input.double_grad() == std::vector<double>{0, 12, 6, 18, 0}, "Double indexed view derivative differs");
                check(input.double_data() == std::vector<double>{123, 3, 1, 2, 456}, "Double indexing mutated source storage");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double indexing pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        raw_operations(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Double indexing leaked storage");
        std::cout << "Float64 indexing: " << checks << " passed, 0 failed\n";
}

}  // namespace indexing

namespace loss {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double loss operation was accepted");
}
void near(const std::vector<double>& values, const std::vector<double>& expected) {
    check(values.size() == expected.size(), "Double loss result size differs");
    for (size_t i = 0; i < values.size(); ++i)
        check(std::abs(values[i] - expected[i]) < 2e-14, "Double loss result differs");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void raw_operations(Backend backend) {
    Tensor x = Tensor::zeros({2,3}, backend, false, DType::Float64);
    Tensor y = Tensor::ones({2,3}, backend, false, DType::Float64);
    Tensor ids = Tensor::from_int_data({2}, {0,2}, backend);
    Tensor g = Tensor::from_double_data({}, {3}, backend);
    struct Case { const char* name; Shape shape; std::vector<Tensor> inputs; std::vector<double> expected; };
    const std::vector<Case> cases{
        {"mse_loss", {}, {x,y}, {1}},
        {"cross_entropy", {}, {x,ids}, {std::log(3.)}},
        {"cross_entropy_grad", {2,3}, {x,ids,g}, {-1,.5,.5,.5,.5,-1}}
    };
    for (const Case& test : cases) {
        Tensor output = Tensor::zeros(test.shape, backend, false, DType::Float64);
        const KernelParams valid = descriptor(output,test.inputs);
        std::vector<BufferPtr> inputs;
        for (const elman::Tensor& input : test.inputs) inputs.push_back(input.node()->value);
        backend_dispatch(test.name,output.node()->value,inputs,valid);
        near(output.double_data(),test.expected);
        const std::vector<uint8_t> before = output.raw_data();
        const int a = KernelParams::input_slot(KernelInput::A), b = KernelParams::input_slot(KernelInput::B);
        for (std::pair<int, int> change : std::vector<std::pair<int,int>>{
                {a,-1},{a,9},{a,1},{a+KernelParams::kInputShapeOffset,-1},{a+KernelParams::kInputShapeOffset,3},
                {a+KernelParams::kInputStrideOffset,1},{b,-1},{b,9},{b+KernelParams::kInputStrideOffset,0},
                {KernelParams::kOutputCountSlot,0},{KernelParams::kOutputRankSlot,-1},{KernelParams::kOutputRankSlot,9},
                {KernelParams::kInputDTypeSlot,int(DType::Float32)}, {KernelParams::kOutputDTypeSlot,int(DType::Float32)},
                {KernelParams::input_slot(KernelInput::D),0}}) {
            KernelParams p = valid; p.v[change.first] = change.second;
            rejects([&] { backend_dispatch(test.name,output.node()->value,inputs,p); });
            check(output.raw_data() == before, "Rejected Double loss descriptor changed output");
        }
        rejects([&] { backend_dispatch(test.name,output.node()->value,{},valid); });
        std::vector<BufferPtr> extra = inputs; extra.push_back(x.node()->value);
        rejects([&] { backend_dispatch(test.name,output.node()->value,extra,valid); });
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<BufferPtr> missing = inputs; missing[i].reset();
            rejects([&] { backend_dispatch(test.name,output.node()->value,missing,valid); });
            std::vector<Tensor> wrong = test.inputs; wrong[i] = cast(wrong[i],DType::Float32);
            rejects([&] { dispatch_tensor_into(test.name,output,wrong); });
        }
        Tensor root = Tensor::zeros({int(x.numel()+output.numel()+1)},backend,false,DType::Float64);
        std::vector<Tensor> operands = test.inputs;
        operands[0] = contiguous_view(root,x.shape());
        rejects([&] { dispatch_tensor_into(test.name,contiguous_view(root,test.shape),operands); });
        rejects([&] { dispatch_tensor_into(test.name,contiguous_view(root,test.shape,1),operands); });
        Tensor disjoint = contiguous_view(root,test.shape,x.numel());
        dispatch_tensor_into(test.name,disjoint,operands);
        near(disjoint.double_data(),test.expected);
        for (size_t i = 1; i < operands.size(); ++i) if (operands[i].dtype() == DType::Float64) {
            Tensor shared = Tensor::zeros({int(operands[i].numel()+output.numel())},backend,false,DType::Float64);
            std::vector<Tensor> alias = test.inputs; alias[i] = contiguous_view(shared,alias[i].shape());
            rejects([&] { dispatch_tensor_into(test.name,contiguous_view(shared,test.shape),alias); });
        }
        if (backend == Backend::CUDA) {
            operands = test.inputs; operands[0] = operands[0].to(Backend::CPU);
            rejects([&] { dispatch_tensor_into(test.name,output,operands); });
        }
    }
    for (int index : {-100,-1,3,INT32_MAX}) {
        Tensor invalid = Tensor::from_int_data({2},{0,index},backend);
        rejects([&] { cross_entropy(x,invalid); });
        rejects([&] { dispatch_tensor("cross_entropy",{},{x,invalid},{},DType::Float64); });
        rejects([&] { dispatch_tensor("cross_entropy_grad",x.shape(),{x,invalid,g},{},DType::Float64); });
    }
    rejects([&] { cross_entropy(x,Tensor::from_int64_data({2},{0,2},backend)); });
    near(cross_entropy(x,reshape(ids,{1,2,1,1})).double_data(),{std::log(3.)});
    rejects([&] { cross_entropy(x,Tensor::from_int_data({1},{0},backend)); });
    rejects([&] { dispatch_tensor("cross_entropy",{1},{x,ids},{},DType::Float64); });
    rejects([&] { dispatch_tensor("cross_entropy",{},{x,reshape(ids,{1,2})},{},DType::Float64); });
    rejects([&] { dispatch_tensor("cross_entropy_grad",x.shape(),{x,ids,reshape(g,{1})},{},DType::Float64); });
    rejects([&] { dispatch_tensor("mse_loss",{},{x,reshape(y,{3,2})},{},DType::Float64); });
    Tensor empty = Tensor::zeros({0,3},backend,true,DType::Float64);
    Tensor empty_ids = Tensor::from_int_data({0},{},backend);
    Tensor empty_loss = cross_entropy(empty,empty_ids);
    check(std::isnan(empty_loss.double_data()[0]), "Empty Double cross entropy must be NaN");
    empty_loss.backward();
    check(empty.double_grad().empty(), "Empty Double cross entropy gradient differs");
    if (backend == Backend::CUDA) rejects([&] { cross_entropy(empty,empty_ids.to(Backend::CPU)); });
    Tensor output = Tensor::zeros({},backend,false,DType::Float64);
    bool limited = false;
    {
        MemoryLimit limit(Backend::CPU,memory_stats(Backend::CPU).live_bytes()+ids.numel()*sizeof(int32_t)-1);
        try { dispatch_tensor_into("cross_entropy",output,{x,ids}); } catch (const MemoryLimitExceeded&) { limited = true; }
    }
    check(limited, "Double loss index validation bypassed the host memory budget");
}
void execution(Backend backend) {
    for (bool pooled : {false,true}) for (bool asynchronous : {false,true}) {
        AllocationPool pool(backend,0,pooled);
        ExecutionStream producer(backend),consumer(backend);
        for (int iteration = 0; iteration < 3; ++iteration) {
            Tensor root,target,output;
            {
                StreamScope scope(producer,asynchronous,true);
                root = Tensor::from_double_data({8},{77,0,0,0,0,0,0,88},backend,true);
                target = Tensor::from_int_data({4},{77,0,2,88},backend);
                output = cross_entropy(contiguous_view(root,{2,3},1),contiguous_view(target,{2},1));
            }
            {
                StreamScope scope(consumer,asynchronous,true);
                near(output.double_data(),{std::log(3.)});
                output.backward(Tensor::from_double_data({},{3},backend));
                near(root.double_grad(),{0,-1,.5,.5,.5,.5,-1,0});
                check(root.double_data() == std::vector<double>{77,0,0,0,0,0,0,88} &&
                      target.int_data() == std::vector<int32_t>{77,0,2,88}, "Double loss mutated its operands");
                Tensor a = Tensor::from_double_data({4},{77,1,2,88},backend,true);
                Tensor b = Tensor::from_double_data({4},{33,3,4,44},backend,true);
                Tensor value = mse_loss(contiguous_view(a,{2},1),contiguous_view(b,{2},1));
                near(value.double_data(),{4});
                value.backward(Tensor::from_double_data({},{3},backend));
                near(a.double_grad(),{0,-6,-6,0}); near(b.double_grad(),{0,6,6,0});
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double loss pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        raw_operations(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before,"Double loss leaked storage");
        std::cout << "Float64 loss: " << checks << " passed, 0 failed\n";
}

}  // namespace loss

namespace matrix {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double matrix operation was accepted");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void malformed(Backend backend) {
    Tensor a = Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend);
    Tensor b = Tensor::from_double_data({3, 2}, {1, 0, 0, 1, 1, 1}, backend);
    Tensor output = Tensor::zeros({2, 2}, backend, false, DType::Float64);
    const KernelParams valid = descriptor(output, {a, b});
    const std::vector<BufferPtr> inputs{a.node()->value, b.node()->value};
    backend_dispatch("matmul", output.node()->value, inputs, valid);
    check(output.double_data() == std::vector<double>{4, 5, 10, 11}, "Raw Double matrix differs");
    const std::vector<uint8_t> before = output.raw_data();
    const int slot = KernelParams::input_slot(KernelInput::A);
    for (std::pair<int, int> bad : std::vector<std::pair<int, int>>{
            {slot, -1}, {slot, 0}, {slot, 9}, {slot + KernelParams::kInputShapeOffset, -1},
            {slot + KernelParams::kInputShapeOffset, 3}, {slot + KernelParams::kInputStrideOffset, 1},
            {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, -1},
            {KernelParams::kOutputRankSlot, 9}, {KernelParams::kOutputShapeSlot, 3},
            {KernelParams::kOutputDTypeSlot, int(DType::Float32)}, {KernelParams::kInputDTypeSlot, int(DType::Float32)},
            {KernelParams::kAttributeSlot, -1}, {KernelParams::kAttributeSlot, 2}}) {
        KernelParams p = valid; p.v[bad.first] = bad.second;
        rejects([&] { backend_dispatch("matmul", output.node()->value, inputs, p); });
        check(output.raw_data() == before, "Rejected Double matrix changed output");
    }
    rejects([&] { backend_dispatch("matmul", output.node()->value, {}, valid); });
    rejects([&] { backend_dispatch("matmul", output.node()->value, {inputs[0]}, valid); });
    rejects([&] { backend_dispatch("matmul", output.node()->value, {inputs[0], inputs[1], inputs[0]}, valid); });
    rejects([&] { backend_dispatch("bmm", output.node()->value, inputs, valid); });
    rejects([&] { backend_dispatch("addmm", output.node()->value, inputs, valid); });
    rejects([&] { matmul(a, cast(b, DType::Float32)); });
    rejects([&] { packed_linear(a, transpose(b, 0, 1), {1, 2}); });
    rejects([&] { packed_linear(a, transpose(b, 0, 1), {}); });
    rejects([&] { packed_linear(a, transpose(b, 0, 1), {-1, 3}); });
    rejects([&] { linear(a, b); });
    rejects([&] { linear(a, transpose(b, 0, 1), Tensor::ones({3}, backend, false, DType::Float64)); });
    if (backend == Backend::CUDA)
        rejects([&] { matmul(a, Tensor::from_double_data({3, 2}, {1, 0, 0, 1, 1, 1}, Backend::CPU)); });
}
void aliases(Backend backend) {
    Tensor root = Tensor::from_double_data({10}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0}, backend);
    Tensor a = contiguous_view(root, {2, 3});
    Tensor b = Tensor::from_double_data({3, 2}, {1, 0, 0, 1, 1, 1}, backend);
    const std::vector<uint8_t> before = root.raw_data();
    for (size_t offset : {size_t(0), size_t(1), size_t(5)})
        rejects([&] { dispatch_tensor_into("matmul", contiguous_view(root, {2, 2}, offset), {a, b}); });
    check(root.raw_data() == before, "Rejected Double matrix alias changed storage");
    Tensor output = contiguous_view(root, {2, 2}, 6);
    dispatch_tensor_into("matmul", output, {a, b});
    check(output.double_data() == std::vector<double>{4, 5, 10, 11}, "Disjoint Double matrix aliases differ");
}
void scalars(Backend backend) {
    Tensor one = Tensor::from_double_data({1, 1}, {1}, backend, true);
    const double fine = 1. + std::ldexp(1., -40);
    check(addmm(one, one, one, 0., fine).double_data()[0] == fine, "Double alpha was rounded");
    check(addmm(one, one, one, fine, 0.).double_data()[0] == fine, "Double beta was rounded");
    Tensor x = Tensor::from_double_data({1, 1}, {1. + std::ldexp(1., -30)}, backend, true);
    Tensor y = matmul(x, one);
    check(y.double_data() == x.double_data(), "Double matrix mantissa was lost");
    y.backward();
    check(x.double_grad() == std::vector<double>{1}, "Double scalar matrix gradient differs");
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    Tensor bad = Tensor::from_double_data({1, 1}, {nan}, backend);
    check(addmm(bad, bad, one, 0., 0.).double_data()[0] == 0., "Zero coefficients consumed NaN inputs");
    check(addmm(bad, one, one, 0., 1.).double_data()[0] == 1., "Zero beta consumed NaN addend");
    Tensor a = Tensor::from_double_data({1, 0}, {}, backend), b = Tensor::from_double_data({0, 1}, {}, backend);
    for (double alpha : {nan, inf, -inf}) {
        check(addmm(bad, a, b, 0., alpha).double_data()[0] == 0., "Empty contraction consumed nonfinite alpha");
        check(addmm(one, a, b, fine, alpha).double_data()[0] == fine, "Empty contraction lost Double beta");
    }
    check(std::isnan(addmm(one, one, one, 0., nan).double_data()[0]), "Nonempty contraction lost NaN alpha");
}
void view_gradients(Backend backend) {
    Tensor source = Tensor::from_double_data({3}, {1. + std::ldexp(1., -40), -2., 3.}, backend);
    Tensor output = Tensor::zeros({8}, backend, false, DType::Float64);
    dispatch_tensor_into("slice_grad", output, {source}, {0, 1, 2});
    check(output.double_data() == std::vector<double>{0., 1. + std::ldexp(1., -40), 0., -2., 0., 3., 0., 0.},
          "Double slice gradient did not preserve exact values");
    const std::vector<uint8_t> before = output.raw_data();
    for (std::vector<int> attributes : std::vector<std::vector<int>>{{0, -1, 1}, {0, 9, 1}, {0, 1, 0}, {0, 1, -1}, {0, 1, 4}, {1, 1, 2}}) {
        rejects([&] { dispatch_tensor_into("slice_grad", output, {source}, attributes); });
        check(output.raw_data() == before, "Rejected Double slice gradient changed output");
    }
    rejects([&] { dispatch_tensor_into("slice_grad", source, {source}, {0, 0, 1}); });
    dispatch_tensor_into("slice_grad", output, {Tensor::from_double_data({0}, {}, backend)}, {0, 8, 1});
    check(output.double_data() == std::vector<double>(8), "Empty Double slice gradient did not zero output");
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 4; ++iteration) {
            Tensor value, input, bias;
            {
                StreamScope scope(producer, asynchronous, true);
                // Offset views exercise the cuBLASLt alignment contract and
                // packed workspace lifetime under independent stream leases.
                input = contiguous_view(Tensor::from_double_data({8}, {0, 1, 2, 3, 4, 5, 6, 0}, backend, true), {2, 3}, 1);
                Tensor weight = contiguous_view(Tensor::from_double_data({8}, {0, 1, 0, 1, 0, 1, 1, 0}, backend, true), {2, 3}, 1);
                bias = contiguous_view(Tensor::from_double_data({4}, {0, .25, -.125, 0}, backend, true), {2}, 1);
                value = packed_linear(input, weight, iteration % 2 ? std::vector<int>{1, 0, 1} : std::vector<int>{0, 2, 0}, bias);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                const std::vector<double> expected = iteration % 2 ? std::vector<double>{4.25, 10.25, 4.875, 10.875} :
                    std::vector<double>{4.25, 4.875, 10.25, 10.875};
                check(value.double_data() == expected, "Double packed matrix layout or stream differs");
                sum_all(value).backward();
                check(input.double_grad() == std::vector<double>{1, 1, 2, 1, 1, 2}, "Double matrix backward lost its producer");
                check(bias.double_grad() == std::vector<double>{2, 2}, "Double packed bias gradient differs");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double matrix pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        malformed(backend); aliases(backend); scalars(backend); view_gradients(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Double matrices leaked storage");
        std::cout << "Float64 matrix: " << checks << " passed, 0 failed\n";
}

}  // namespace matrix

namespace normalization {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double normalization operation was accepted");
}
Tensor normalize(int kind, const Tensor& x, const Tensor& w, const Tensor& b, double eps) {
    if (kind == 0) return layer_norm(x, w, b, eps);
    if (kind == 1) return rms_norm(x, w, eps);
    if (kind == 2) return batch_norm(x, w, b, eps);
    return group_norm(x, kind == 3 ? 1 : 3, w, b, eps);
}
void invalid(Backend backend) {
    Tensor x = Tensor::ones({2, 3}, backend, false, DType::Float64);
    Tensor w = Tensor::ones({3}, backend, false, DType::Float64);
    Tensor bad = Tensor::ones({4}, backend, false, DType::Float64);
    for (int kind = 0; kind < 5; ++kind) {
        rejects([&] { normalize(kind, x, bad, w, 1e-5); });
        rejects([&] { normalize(kind, x, w, w, -1.); });
        rejects([&] { normalize(kind, x, w, w, std::numeric_limits<double>::quiet_NaN()); });
        if (kind != 1) {
            rejects([&] { normalize(kind, x, w, bad, 1e-5); });
            rejects([&] { normalize(kind, x, cast(w, DType::Float32), w, 1e-5); });
        }
        if (backend == Backend::CUDA) {
            Tensor cpu = w.to(Backend::CPU);
            rejects([&] { normalize(kind, x, cpu, w, 1e-5); });
        }
    }
    for (int groups : {0, -1, 2, 4}) rejects([&] { group_norm(x, groups, w, w, 1e-5); });
    rejects([&] { batch_norm(reshape(w, {1, 3}), w, w); });
    rejects([&] { group_norm(reshape(w, {1, 3}), 3, w, w); });
    rejects([&] { rms_norm(x, w, 1e-5, RmsNormRounding(9)); });
    rejects([&] { rms_norm(x, cast(w, DType::Float32), 1e-5, RmsNormRounding::Normalized); });
}
void fused_arithmetic(Backend backend) {
    Tensor a = Tensor::from_double_data({2, 3}, {1. + 0x1p-27, 0x1p-1022, -0., -1. - 0x1p-27, 0x1p100, 3.}, backend);
    Tensor b = Tensor::from_double_data({1, 3}, {1. - 0x1p-27, .5, -2.}, backend);
    Tensor c = Tensor::from_double_data({}, {-1.}, backend);
    Tensor result = dispatch_tensor("float64_fma", {2, 3}, {a, b, c}, {}, DType::Float64);
    std::vector<double> av = a.double_data(), bv = b.double_data(), actual = result.double_data();
    for (size_t i = 0; i < actual.size(); ++i)
        check(actual[i] == std::fma(av[i], bv[i % 3], -1.), "Double fused multiply-add differs");
    check(actual[0] == -0x1p-54, "Double fused multiply-add rounded the product separately");
    dispatch_tensor_into("float64_fma", a, {a, b, c});
    check(a.double_data() == actual, "In-place Double fused multiply-add differs");
    Tensor root = Tensor::zeros({7}, backend, false, DType::Float64);
    Tensor input = contiguous_view(root, {2, 3});
    const std::vector<uint8_t> before = root.raw_data();
    rejects([&] { dispatch_tensor_into("float64_fma", contiguous_view(root, {2, 3}, 1), {input, b, c}); });
    check(root.raw_data() == before, "Rejected fused multiply-add changed storage");
    rejects([&] { dispatch_tensor("float64_fma", {2, 3}, {a, b}, {}, DType::Float64); });
    rejects([&] { dispatch_tensor("float64_fma", {2, 3}, {a, b, c, c}, {}, DType::Float64); });
    rejects([&] { dispatch_tensor("float64_fma", {2, 3}, {a, b, cast(c, DType::Int32)}, {}, DType::Float64); });
    rejects([&] { dispatch_tensor("float64_fma", {2, 3}, {a, b, c}, {}, DType::Float32); });
    rejects([&] { dispatch_tensor("float64_fma", {3, 2}, {a, b, c}, {}, DType::Float64); });
    Tensor empty = Tensor::zeros({0, 3}, backend, false, DType::Float64);
    check(dispatch_tensor("float64_fma", {0, 3}, {empty, b, c}, {}, DType::Float64).numel() == 0,
          "Empty Double fused multiply-add differs");
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 3; ++iteration) for (int kind = 0; kind < 5; ++kind) {
            Tensor value, input, weight, bias;
            {
                StreamScope scope(producer, asynchronous, true);
                input = Tensor::zeros({8}, backend, true, DType::Float64);
                weight = Tensor::from_double_data({5}, {77, 2, 2, 2, 88}, backend, true);
                bias = Tensor::from_double_data({5}, {33, 1, 2, 3, 44}, backend, true);
                value = normalize(kind, contiguous_view(input, {2, 3}, 1), contiguous_view(weight, {3}, 1),
                                  contiguous_view(bias, {3}, 1), 1.);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                check(value.double_data() == (kind == 1 ? std::vector<double>(6, 0.) : std::vector<double>{1, 2, 3, 1, 2, 3}),
                      "Double normalization stream dependency differs");
                value.backward(Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend));
                std::vector<double> dx = kind == 1 ? std::vector<double>{2, 4, 6, 8, 10, 12} :
                    kind == 2 ? std::vector<double>{-3, -3, -3, 3, 3, 3} :
                    kind == 4 ? std::vector<double>(6, 0.) : std::vector<double>{-2, 0, 2, -2, 0, 2};
                dx.insert(dx.begin(), 0.); dx.push_back(0.);
                check(input.double_grad() == dx, "Double normalization view input gradient differs");
                check(weight.double_grad() == std::vector<double>(5, 0.), "Double normalization view weight gradient differs");
                check(bias.double_grad() == (kind == 1 ? std::vector<double>{} : std::vector<double>{0, 5, 7, 9, 0}),
                      "Double normalization view bias gradient differs");
                check(input.double_data() == std::vector<double>(8, 0.) &&
                      weight.double_data() == std::vector<double>{77, 2, 2, 2, 88} &&
                      bias.double_data() == std::vector<double>{33, 1, 2, 3, 44}, "Double normalization mutated input storage");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double normalization did not reuse pooled storage");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        invalid(backend); fused_arithmetic(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Double normalization leaked storage");
        std::cout << "Float64 normalization: " << checks << " passed, 0 failed\n";
}

}  // namespace normalization

namespace optimizer {

using namespace elman;
int checks=0;
void check(bool value,const char* message) {++checks;if (!value) throw std::runtime_error(message);}
template<class F> void rejects(F action) {
    bool rejected=false;try {action();} catch (const std::exception&) {rejected=true;}
    check(rejected,"Invalid Double optimizer operation was accepted");
}
class Probe final:public Model {
public:
    Probe(Backend backend,bool mixed=false,bool narrow=false):Model("DoubleProbe","probe",backend,1) {
        Tensor scalar=narrow ? Tensor::from_data({}, {1.f},backend,true) : Tensor::from_double_data({}, {1.+std::ldexp(1.,-40)},backend,true);
        Tensor values=narrow ? Tensor::from_data({3},{1.f,2.f,-3.f},backend,true) : Tensor::from_double_data({3},{1e100,1e-100,-.3},backend,true);
        parameters_.push_back({"scalar",scalar});parameters_.push_back({"values",values});
        parameters_.push_back({"empty",Tensor::zeros({2,0,3},backend,true,narrow ? DType::Float32 : DType::Float64)});
        if (mixed) parameters_.push_back({"narrow",Tensor::from_data({2},{.1f,-.2f},backend,true)});
    }
    Tensor forward(const Tensor& input,bool,uint32_t) override {return input;}
    Shape sample_shape() const override {return {1};}
    int output_classes() const override {return 1;}
};
Json settings(const std::string& algorithm,double clip=.7) {
    return Json::Object{{"optimizer",algorithm},{"learning_rate",.01},{"beta1",.9},{"beta2",.999},
        {"epsilon",1e-8},{"momentum",.6},{"weight_decay",.01},{"clip_grad_norm",clip}};
}
void gradients(Probe& model,int step) {
    for (size_t i=0;i<model.parameters().size();++i) {
        Tensor& x=model.parameters()[i].value;x.zero_grad();
        const double value=(step%2 ? -1. : 1.)*(.3+std::ldexp(1.,-40))*(i+1);
        if (x.dtype()==DType::Float64) x.backward(Tensor::from_double_data(x.shape(),std::vector<double>(x.numel(),value),x.backend()));
        else x.backward(Tensor::from_data(x.shape(),std::vector<float>(x.numel(),float(value)),x.backend()));
    }
}
using Snapshot=std::vector<std::vector<uint8_t>>;
Snapshot snapshot(const Probe& model,const TrainingState& state) {
    Snapshot result;
    std::vector<uint8_t> metadata(sizeof(state.step));std::memcpy(metadata.data(),&state.step,sizeof(state.step));result.push_back(metadata);
    result.emplace_back(state.optimizer.begin(),state.optimizer.end());
    for (const elman::Parameter& parameter:model.parameters()) result.push_back(parameter.value.raw_data());
    for (const std::vector<std::vector<float>>* group:{&state.first_moment,&state.second_moment}) for (const std::vector<float>& row:*group) {
        std::vector<uint8_t> raw(row.size()*sizeof(float));if (!raw.empty()) std::memcpy(raw.data(),row.data(),raw.size());result.push_back(raw);
    }
    for (const std::vector<elman::Tensor>* group:{&state.first_moment_double,&state.second_moment_double})
        for (const elman::Tensor& row:*group) result.push_back(row.defined() ? row.raw_data() : std::vector<uint8_t>{});
    return result;
}
std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
void write_bytes(const std::filesystem::path& path,const std::vector<uint8_t>& bytes) {
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
    check(bool(file),"Cannot write optimizer checkpoint fixture");
}
void checksum(std::vector<uint8_t>& bytes) {
    uint64_t hash=14695981039346656037ull;
    for (size_t i=0;i<bytes.size()-8;++i) {hash^=bytes[i];hash*=1099511628211ull;}
    for (int i=0;i<8;++i) bytes[bytes.size()-8+i]=uint8_t(hash>>(8*i));
}
std::vector<size_t> dtype_offsets(const std::vector<uint8_t>& bytes) {
    size_t cursor=12;
    LocalFunction u32{[&]() {uint32_t result=0;for (int i=0;i<4;++i) result|=uint32_t(bytes.at(cursor++))<<(8*i);return result;}};
    LocalFunction u64{[&]() {uint64_t result=0;for (int i=0;i<8;++i) result|=uint64_t(bytes.at(cursor++))<<(8*i);return result;}};
    LocalFunction string{[&]() {uint32_t length=u32();cursor+=length;}};
    string();string();u64();string();const uint32_t count=u32();
    std::vector<size_t> offsets;
    for (uint32_t i=0;i<count;++i) {
        string();uint32_t rank=u32();cursor+=4*rank;offsets.push_back(cursor);
        uint32_t dtype=u32();
        for (int j=0;j<3;++j) {uint64_t length=u64();cursor+=size_t(length)*(dtype==uint32_t(DType::Float64) ? 8 : 4);}
    }
    check(cursor==bytes.size()-8,"Invalid checkpoint fixture layout");return offsets;
}
void resume(Backend backend,const std::filesystem::path& directory) {
    const Json definition=Json::Object{{"architecture","probe"},{"version",2}};
    for (const std::string algorithm:{"sgd","adam","adamw"}) for (bool mixed:{false,true}) {
        Probe model(backend,mixed),restored(backend,mixed);
        Optimizer optimizer(settings(algorithm),model),restored_optimizer(settings(algorithm),restored);
        for (int step=0;step<3;++step) {gradients(model,step);optimizer.update(model,.01);}
        const std::filesystem::path path=directory/(algorithm+(mixed ? "-mixed" : "-double")+".bin");
        save_checkpoint(path.string(),model,definition,optimizer.state());
        std::vector<uint8_t> bytes=read_bytes(path);check(bytes.at(8)==2,"Double checkpoint did not use version 2");
        load_checkpoint(path.string(),restored,definition,&restored_optimizer.state());
        check(snapshot(model,optimizer.state())==snapshot(restored,restored_optimizer.state()),"Double checkpoint lost values or moments");
        for (size_t i=0;i<model.parameters().size();++i) if (model.parameters()[i].value.dtype()==DType::Float64) {
            const TrainingState& state=restored_optimizer.state();
            check(state.first_moment[i].empty() && state.second_moment[i].empty(),"Double moments used Float32 state");
            for (const elman::Tensor* moment:{&state.first_moment_double[i],&state.second_moment_double[i]})
                check(moment->dtype()==DType::Float64 && moment->backend()==backend && !moment->requires_grad(),"Double moment lost its backend or dtype");
        }
        if (backend==Backend::CUDA) {
            Probe cpu(Backend::CPU,mixed);Optimizer cpu_optimizer(settings(algorithm),cpu);
            load_checkpoint(path.string(),cpu,definition,&cpu_optimizer.state());
            check(snapshot(model,optimizer.state())==snapshot(cpu,cpu_optimizer.state()),"Cross-backend checkpoint differs");
            const std::filesystem::path cross=directory/(algorithm+(mixed ? "-mixed-cpu" : "-double-cpu")+".bin");
            save_checkpoint(cross.string(),cpu,definition,cpu_optimizer.state());
            check(read_bytes(cross)==bytes,"Cross-backend checkpoint bytes differ");
        }
        for (int step=3;step<8;++step) {
            gradients(model,step);gradients(restored,step);
            TrainingState old_state=optimizer.state();std::vector<uint8_t> old_first=old_state.first_moment_double[0].raw_data();
            uint64_t before=backend_dispatch_count(backend);
            check(optimizer.update(model,.01)==restored_optimizer.update(restored,.01),"Resumed gradient norm differs");
            check(backend_dispatch_count(backend)>before,"Double optimizer did not use native kernals");
            check(old_state.first_moment_double[0].raw_data()==old_first,"Optimizer mutated a saved state snapshot");
            check(snapshot(model,optimizer.state())==snapshot(restored,restored_optimizer.state()),"Resumed Double optimizer differs");
        }
        // Loading failures must preserve all parameters and the caller's state.
        const Snapshot before=snapshot(restored,restored_optimizer.state());
        const std::filesystem::path bad=directory/(algorithm+(mixed ? "-mixed-bad" : "-bad")+".bin");
        for (size_t cut:{size_t(0),size_t(1),size_t(7),size_t(12),size_t(30),bytes.size()/2,bytes.size()-1}) {
            write_bytes(bad,std::vector<uint8_t>(bytes.begin(),bytes.begin()+cut));
            rejects([&]{load_checkpoint(bad.string(),restored,definition,&restored_optimizer.state());});
            check(snapshot(restored,restored_optimizer.state())==before,"Truncated checkpoint partially mutated training state");
        }
        for (size_t offset:{size_t(0),size_t(8),bytes.size()-1}) {
            std::vector<uint8_t> corrupt=bytes;corrupt[offset]^=0x40;write_bytes(bad,corrupt);
            rejects([&]{load_checkpoint(bad.string(),restored,definition,&restored_optimizer.state());});
            check(snapshot(restored,restored_optimizer.state())==before,"Corrupt checkpoint partially mutated training state");
        }
        for (size_t offset:dtype_offsets(bytes)) {
            std::vector<uint8_t> corrupt=bytes;corrupt[offset]=255;checksum(corrupt);write_bytes(bad,corrupt);
            rejects([&]{load_checkpoint(bad.string(),restored,definition,&restored_optimizer.state());});
            check(snapshot(restored,restored_optimizer.state())==before,"Dtype mismatch partially mutated training state");
        }
        std::vector<uint8_t> corrupt=bytes;corrupt.push_back(0);write_bytes(bad,corrupt);
        rejects([&]{load_checkpoint(bad.string(),restored,definition,&restored_optimizer.state());});
        check(snapshot(restored,restored_optimizer.state())==before,"Trailing checkpoint bytes partially mutated training state");
        elman::Json::Object wrong=definition.object();wrong["version"]=3;
        rejects([&]{load_checkpoint(path.string(),restored,Json(wrong),&restored_optimizer.state());});
        check(snapshot(restored,restored_optimizer.state())==before,"Wrong definition partially mutated state");
        TrainingState invalid=optimizer.state();invalid.first_moment_double[0]=cast(invalid.first_moment_double[0],DType::Float32);
        rejects([&]{save_checkpoint(path.string(),model,definition,invalid);});
        check(read_bytes(path)==bytes,"Invalid state replaced a valid checkpoint");
        const std::vector<uint8_t> saved=model.parameters()[0].value.raw_data();
        model.parameters()[0].value.assign_double({INFINITY});
        rejects([&]{save_checkpoint(path.string(),model,definition,optimizer.state());});
        check(read_bytes(path)==bytes && !std::filesystem::exists(path.string()+".tmp"),"Failed save damaged checkpoint or left temporary output");
        model.parameters()[0].value.assign_raw(saved);
    }
    Probe narrow(backend,false,true),wide(backend);Optimizer narrow_optimizer(settings("sgd"),narrow),wide_optimizer(settings("sgd"),wide);
    gradients(narrow,0);narrow_optimizer.update(narrow,.01);
    const std::filesystem::path legacy=directory/"legacy.bin";save_checkpoint(legacy.string(),narrow,definition,narrow_optimizer.state());
    check(read_bytes(legacy).at(8)==1,"Legacy checkpoint encoding changed");
    const Snapshot before=snapshot(wide,wide_optimizer.state());
    rejects([&]{load_checkpoint(legacy.string(),wide,definition,&wide_optimizer.state());});
    check(snapshot(wide,wide_optimizer.state())==before,"Legacy checkpoint silently narrowed a Double model");
}
void validation(Backend backend) {
    for (const std::string algorithm:{"sgd","adam","adamw"}) {
        Probe model(backend);Optimizer optimizer(settings(algorithm,0.),model);gradients(model,0);
        for (double rate:{-1.,double(NAN),double(INFINITY)}) {
            Snapshot before=snapshot(model,optimizer.state());
            rejects([&]{optimizer.update(model,rate);});check(snapshot(model,optimizer.state())==before,"Invalid rate mutated optimizer");
        }
        for (int kind=0;kind<7;++kind) {
            gradients(model,0);TrainingState saved=optimizer.state();
            Tensor& value=model.parameters()[1].value;
            if (kind==0) {value.zero_grad();value.backward(Tensor::from_double_data({3},{NAN,1.,2.},backend));}
            if (kind==1) optimizer.state().second_moment_double[1]=Tensor::from_double_data({3},{-1.,0.,0.},backend);
            if (kind==2) optimizer.state().first_moment_double[1]=Tensor::zeros({2},backend,false,DType::Float64);
            if (kind==3) optimizer.state().step=uint64_t(std::numeric_limits<int64_t>::max());
            if (kind==4) optimizer.state().optimizer="other";
            if (kind==5) value.zero_grad();
            if (kind==6) {value.zero_grad();value.backward(Tensor::from_double_data({3},{1e308,1.,2.},backend));}
            Snapshot before=snapshot(model,optimizer.state());
            rejects([&]{optimizer.update(model,kind==6 && algorithm=="sgd" ? 1e100 : .01);});
            check(snapshot(model,optimizer.state())==before,"Rejected update partially mutated parameters or state");
            optimizer.state()=saved;
        }
        gradients(model,0);Tensor graph=mul(model.parameters()[0].value,2.);
        optimizer.update(model,.01);rejects([&]{graph.backward();});
        check(model.parameters()[0].value.requires_grad(),"Optimizer detached the parameter leaf");
    }
    Probe model(backend);elman::Json::Object config=settings("adam",0.).object();config["weight_decay"]=0;config["epsilon"]=1e-100;
    Optimizer optimizer(Json(config),model);
    for (elman::Parameter& p:model.parameters()) {p.value.zero_grad();p.value.backward(Tensor::from_double_data(p.value.shape(),std::vector<double>(p.value.numel(),1e-100),backend));}
    optimizer.update(model,.01);
    const double first=optimizer.state().first_moment_double[0].double_data()[0];
    const double second=optimizer.state().second_moment_double[0].double_data()[0];
    check(std::abs(first/1e-101-1.)<1e-14 && std::abs(second/1e-203-1.)<1e-13,"Double moments lost values below Float32 range");
}
void execution(Backend backend) {
    for (bool pooled:{false,true}) for (bool asynchronous:{false,true}) {
        AllocationPool pool(backend,0,pooled);ExecutionStream producer(backend),consumer(backend);
        for (int iteration=0;iteration<3;++iteration) {
            Probe model(backend);Optimizer optimizer(settings("adam"),model);
            {StreamScope scope(producer,asynchronous,true);gradients(model,0);optimizer.update(model,.01);}
            {StreamScope scope(consumer,asynchronous,true);gradients(model,1);optimizer.update(model,.01);}
            check(optimizer.state().step==2,"Asynchronous optimizer step differs");
        }
        producer.synchronize();consumer.synchronize();
        if (pooled) check(pool.stats().reuses>0,"Optimizer did not exercise pool reuse");
    }
}

void run(Backend backend, const std::filesystem::path& directory) {
    check(!std::filesystem::exists(directory),"Optimizer output already exists");std::filesystem::create_directories(directory);
        const size_t before=memory_stats(backend).live_bytes();
        resume(backend,directory);validation(backend);execution(backend);
        check(memory_stats(backend).live_bytes()==before,"Double optimizer leaked storage");
        std::cout << "Float64 optimizer: " << checks << " passed, 0 failed\n";
}

}  // namespace optimizer

namespace pointwise {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Float64 operation was accepted");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void malformed(Backend backend) {
    Tensor a = Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend);
    Tensor b = Tensor::from_double_data({1, 3}, {1, 2, 3}, backend);
    Tensor out = Tensor::zeros({2, 3}, backend, false, DType::Float64);
    const KernelParams valid = descriptor(out, {a, b});
    const std::vector<BufferPtr> inputs = {a.node()->value, b.node()->value};
    backend_dispatch("add", out.node()->value, inputs, valid);
    check(out.double_data() == std::vector<double>{2, 4, 6, 5, 7, 9}, "Raw Float64 broadcast differs");
    const std::vector<uint8_t> before = out.raw_data();
    const int a_slot = KernelParams::input_slot(KernelInput::A), b_slot = KernelParams::input_slot(KernelInput::B);
    for (std::pair<int, int> bad : std::vector<std::pair<int, int>>{
            {a_slot, -1}, {a_slot, 9}, {b_slot, -1}, {b_slot, 9},
            {a_slot + KernelParams::kInputShapeOffset, -1}, {a_slot + KernelParams::kInputShapeOffset, 3},
            {b_slot + KernelParams::kInputShapeOffset, 2},
            {a_slot + KernelParams::kInputStrideOffset, 1}, {b_slot + KernelParams::kInputStrideOffset + 1, 0},
            {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, -1},
            {KernelParams::kOutputRankSlot, 9}, {KernelParams::kOutputShapeSlot, 3},
            {KernelParams::kOutputDTypeSlot, int(DType::Float32)}, {KernelParams::kInputDTypeSlot, int(DType::Float32)}}) {
        KernelParams p = valid; p.v[bad.first] = bad.second;
        rejects([&] { backend_dispatch("add", out.node()->value, inputs, p); });
        check(out.raw_data() == before, "Rejected descriptor mutated output");
    }
    rejects([&] { backend_dispatch("add", out.node()->value, {}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0]}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0], {}}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0], inputs[1], inputs[1]}, valid); });
    Tensor integer = cast(b, DType::Int32);
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0], integer.node()->value}, descriptor(out, {a, integer})); });
    Tensor wrong = Tensor::zeros({2, 3}, backend);
    rejects([&] { backend_dispatch("add", wrong.node()->value, inputs, descriptor(wrong, {a, b})); });
    Tensor empty = Tensor::from_double_data({0}, {}, backend);
    KernelParams p = descriptor(empty, {empty});
    p.v[a_slot] = 3;
    p.v[a_slot + KernelParams::kInputShapeOffset] = INT32_MAX;
    p.v[a_slot + KernelParams::kInputShapeOffset + 1] = INT32_MAX;
    p.v[a_slot + KernelParams::kInputShapeOffset + 2] = 0;
    p.v[a_slot + KernelParams::kInputStrideOffset] = 0;
    p.v[a_slot + KernelParams::kInputStrideOffset + 1] = 0;
    p.v[a_slot + KernelParams::kInputStrideOffset + 2] = 1;
    rejects([&] { backend_dispatch("abs", empty.node()->value, {empty.node()->value}, p); });
}
void values(Backend backend) {
    const std::vector<double> exact = {1. + 0x1p-52, -1. - 0x1p-52, 0x1p-1074, -0., 0x1p100};
    Tensor value = Tensor::from_double_data({5}, exact, backend, true);
    check(value.double_data() == exact, "Typed Double upload lost precision");
    check(std::signbit(value.double_data()[3]), "Typed Double upload lost negative zero");
    check(value.to(Backend::CPU).raw_data() == value.raw_data(), "Double device transfer changed bytes");
    value.assign_double(exact);
    check(value.double_data() == exact, "Typed Double assignment lost precision");
    rejects([&] { value.assign_double({1}); });
    rejects([&] { Tensor::from_double_data({2}, {1}, backend); });
    Tensor narrow = Tensor::ones({2}, backend);
    rejects([&] { narrow.double_data(); });
    rejects([&] { narrow.assign_double({1, 2}); });
    check(value.double_grad().empty(), "Fresh Double leaf has a gradient");
    Tensor x = Tensor::from_double_data({2}, {1. + 0x1p-40, -2. - 0x1p-40}, backend, true);
    mul(x, x).backward(Tensor::ones({2}, backend, false, DType::Float64));
    check(x.double_grad() == std::vector<double>{2. + 0x1p-39, -4. - 0x1p-39}, "Double backward rounded to Float32");
    Tensor scalar = Tensor::from_double_data({}, {1. + 0x1p-40}, backend, true);
    mul(scalar, scalar).backward();
    check(scalar.double_grad() == std::vector<double>{2. + 0x1p-39}, "Default Double backward seed differs");
    check(mul(Tensor::from_double_data({}, {1.}, backend), 1. + 0x1p-40).double_data() ==
          std::vector<double>{1. + 0x1p-40}, "Double scalar parameter lost precision");
    check(add(scalar, 1. + 0x1p-40).double_data() == std::vector<double>{2. + 0x1p-39},
          "Double scalar addition lost precision");
    Tensor host = Tensor::from_data({3}, {1.25f, -2.5f, 0.f}, backend, false, DType::Float64);
    check(host.double_data() == std::vector<double>{1.25, -2.5, 0}, "Float host compatibility upload differs");
    host.assign({2.5f, 1.25f, -1.f});
    check(host.data() == std::vector<float>{2.5f, 1.25f, -1.f}, "Float host compatibility round trip differs");
}
void aliases(Backend backend) {
    Tensor root = Tensor::from_double_data({12}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}, backend);
    Tensor source = contiguous_view(root, {2, 3});
    Tensor ones = Tensor::ones({2, 3}, backend, false, DType::Float64);
    dispatch_tensor_into("add", source, {source, ones});
    check(source.double_data() == std::vector<double>{2, 3, 4, 5, 6, 7}, "Exact in-place Double operation differs");
    Tensor shifted = contiguous_view(root, {2, 3}, 1);
    std::vector<uint8_t> before = root.raw_data();
    rejects([&] { dispatch_tensor_into("add", shifted, {source, ones}); });
    rejects([&] { dispatch_tensor_into("add", source, {contiguous_view(root, {1, 3}), ones}); });
    check(root.raw_data() == before, "Rejected overlap mutated Double storage");
    Tensor disjoint = contiguous_view(root, {2, 3}, 6);
    dispatch_tensor_into("mul", disjoint, {source, source});
    check(disjoint.double_data() == std::vector<double>{4, 9, 16, 25, 36, 49}, "Disjoint Double aliases differ");
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 4; ++iteration) {
            Tensor value;
            {
                StreamScope scope(producer, asynchronous, true);
                Tensor input = Tensor::from_double_data({4}, {1. + 0x1p-40, -2. - 0x1p-40, 0x1p-100, 0x1p100}, backend);
                value = mul(input, Tensor::from_double_data({}, {2}, backend));
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                Tensor result = div(value, 2.f);
                check(result.double_data() == std::vector<double>{1. + 0x1p-40, -2. - 0x1p-40, 0x1p-100, 0x1p100},
                      "Double stream dependencies or precision differ");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double pointwise pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        malformed(backend); values(backend); aliases(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Double pointwise operations leaked storage");
        std::cout << "Float64 pointwise: " << checks << " passed, 0 failed\n";
}

}  // namespace pointwise

namespace random {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double random operation was accepted");
}
void near(const std::vector<double>& values, const std::vector<double>& expected) {
    check(values.size() == expected.size(), "Double random result size differs");
    for (size_t i = 0; i < values.size(); ++i)
        check(std::abs(values[i] - expected[i]) < 2e-14, "Double random result differs");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
int bits(float value) { int result;std::memcpy(&result,&value,sizeof(value));return result; }
std::vector<int> attributes(const std::string& op, int seed, int offset, double first, double second, bool wide=true) {
    std::vector<int> result{seed,offset,bits(float(first)),op=="dropout" ? 1 : bits(float(second))};
    if (wide) {
        result.resize(9);
        std::memcpy(result.data()+4,&first,sizeof(first));std::memcpy(result.data()+6,&second,sizeof(second));result[8]=1;
    }
    return result;
}
void raw_operations(Backend backend) {
    for (const std::string op:{"uniform","normal","dropout"}) {
        const bool dropout_op=op=="dropout";
        Tensor input=Tensor::ones({2,3},backend,false,DType::Float64);
        Tensor output=Tensor::zeros({2,3},backend,false,DType::Float64);
        const std::vector<Tensor> operands=dropout_op ? std::vector<Tensor>{input} : std::vector<Tensor>{};
        const std::vector<int> attrs=attributes(op,17,0,.3,1.);
        KernelParams valid=descriptor(output,operands);
        for (size_t i=0;i<attrs.size();++i) valid.v[KernelParams::kAttributeSlot+i]=attrs[i];
        std::vector<BufferPtr> inputs;
        if (dropout_op) inputs.push_back(input.node()->value);
        backend_dispatch(op,output.node()->value,inputs,valid);
        const std::vector<uint8_t> before=output.raw_data();
        std::vector<std::pair<int,int>> changes{{KernelParams::kOutputCountSlot,0},{KernelParams::kOutputRankSlot,-1},
            {KernelParams::kOutputRankSlot,9},{KernelParams::kOutputRankSlot,0},{KernelParams::kOutputShapeSlot,-1},
            {KernelParams::kOutputShapeSlot,3},{KernelParams::kOutputDTypeSlot,int(DType::Float32)},
            {KernelParams::kAttributeSlot+8,2},{KernelParams::input_slot(KernelInput::D),0}};
        if (dropout_op) {
            const int a=KernelParams::input_slot(KernelInput::A);
            changes.insert(changes.end(),{{a,-1},{a,9},{a,1},{a+KernelParams::kInputShapeOffset,-1},
                {a+KernelParams::kInputShapeOffset,3},{a+KernelParams::kInputStrideOffset,1},
                {KernelParams::kInputDTypeSlot,int(DType::Float32)},{KernelParams::kAttributeSlot+3,2}});
        } else changes.emplace_back(KernelParams::input_slot(KernelInput::A),0);
        for (std::pair<int, int> change:changes) {
            KernelParams p=valid;p.v[change.first]=change.second;
            rejects([&] { backend_dispatch(op,output.node()->value,inputs,p); });
            check(output.raw_data()==before,"Rejected Double random descriptor changed output");
        }
        std::vector<BufferPtr> extra=inputs;extra.push_back(input.node()->value);
        rejects([&] { backend_dispatch(op,output.node()->value,extra,valid); });
        if (dropout_op) {
            rejects([&] { backend_dispatch(op,output.node()->value,{},valid); });
            rejects([&] { dispatch_tensor_into(op,output,{cast(input,DType::Float32)},attrs); });
            Tensor root=Tensor::ones({13},backend,false,DType::Float64);
            Tensor view=contiguous_view(root,{2,3});
            const std::vector<uint8_t> original=root.raw_data();
            rejects([&] { dispatch_tensor_into(op,contiguous_view(root,{2,3},1),{view},attrs); });
            check(root.raw_data()==original,"Rejected Double dropout alias changed storage");
            dispatch_tensor_into(op,view,{view},attrs);
            check(view.raw_data()==before,"In-place Double dropout differs");
            view.assign_double(std::vector<double>(6,1.));
            Tensor disjoint=contiguous_view(root,{2,3},6);
            dispatch_tensor_into(op,disjoint,{view},attrs);
            check(disjoint.raw_data()==before,"Disjoint Double dropout views differ");
            if (backend==Backend::CUDA) rejects([&] { dispatch_tensor_into(op,output,{input.to(Backend::CPU)},attrs); });
        }
        const std::vector<int> invalid=op=="uniform" ? attributes(op,1,0,2.,1.) : op=="normal" ? attributes(op,1,0,0.,-1.) : attributes(op,1,0,1.1,0.);
        rejects([&] { dispatch_tensor_into(op,output,operands,invalid); });
        const double nan=std::numeric_limits<double>::quiet_NaN();
        const std::vector<int> bad=attributes(op,1,0,op=="normal" ? 0. : nan,op=="normal" ? nan : 1.);
        rejects([&] { dispatch_tensor_into(op,output,operands,bad); });
    }
    const double maximum=std::numeric_limits<double>::max();
    rejects([&] { Tensor::uniform({0},-maximum,maximum,1,backend,false,DType::Float64); });
    rejects([&] { Tensor::normal({0},0.,-1.,1,backend,false,DType::Float64); });
    const double precise=1.+std::ldexp(1.,-40);
    check(Tensor::uniform({7},precise,precise,1,backend,false,DType::Float64).double_data()==std::vector<double>(7,precise),"Double uniform rounded its parameters");
    check(Tensor::normal({7},precise,0.,1,backend,false,DType::Float64).double_data()==std::vector<double>(7,precise),"Double normal rounded its parameters");
    Tensor exceptional=Tensor::from_double_data({7},{NAN,INFINITY,-INFINITY,0.,-0.,1.,-1.},backend,true);
    Tensor dropped=dropout(exceptional,1.,3);
    std::vector<double> values=dropped.double_data();
    for (size_t i=0;i<3;++i) check(std::isnan(values[i]),"Double dropout erased nonfinite values");
    for (size_t i=3;i<7;++i) check(values[i]==0. && std::signbit(values[i])==(i==4 || i==6),"Double dropout changed zero signs");
    dropped.backward(Tensor::from_double_data({7},{INFINITY,-INFINITY,NAN,0.,-0.,1.,-1.},backend));
    values=exceptional.double_grad();
    for (size_t i=0;i<3;++i) check(std::isnan(values[i]),"Double dropout erased nonfinite gradients");
    for (size_t i=3;i<7;++i) check(values[i]==0. && std::signbit(values[i])==(i==4 || i==6),"Double dropout gradient changed zero signs");
    check(dropout(exceptional,0.,7).node()->value==exceptional.node()->value &&
          dropout(exceptional,.3,7,false).node()->value==exceptional.node()->value,"Identity Double dropout copied its input");
}
void replay(Backend backend) {
    for (const std::string op:{"uniform","normal","dropout"}) {
        for (int seed:{0,17,-1}) {
            Tensor input=Tensor::ones({257},backend,false,DType::Float64);
            LocalFunction make{[&](int count,int offset,int selected_seed) {
                return dispatch_tensor(op,{count},op=="dropout" ? std::vector<Tensor>{contiguous_view(input,{count})} : std::vector<Tensor>{},
                    attributes(op,selected_seed,offset,op=="dropout" ? .5 : .125,1.75),DType::Float64,backend);
            }};
            Tensor full=make(257,0,seed),small=make(17,0,seed);
            check(full.raw_data()==make(257,0,seed).raw_data(),"Double random replay differs");
            check(small.raw_data()==contiguous_view(full,{17}).raw_data(),"Double random replay depends on launch length");
            check(full.raw_data()!=make(257,1,seed).raw_data(),"Double random stream offset has no effect");
            check(full.raw_data()!=make(257,0,seed^123).raw_data(),"Double random seed has no effect");
            if (backend==Backend::CUDA) {
                Tensor cpu_input=Tensor::ones({257},Backend::CPU,false,DType::Float64);
                Tensor cpu=dispatch_tensor(op,{257},op=="dropout" ? std::vector<Tensor>{cpu_input} : std::vector<Tensor>{},
                    attributes(op,seed,0,op=="dropout" ? .5 : .125,1.75),DType::Float64,Backend::CPU);
                if (op=="normal") near(full.double_data(),cpu.double_data());
                else check(full.raw_data()==cpu.raw_data(),"Double random CPU/CUDA counters differ");
            }
        }
    }
    std::vector<double> values=Tensor::uniform({4096},0.,1.,13,backend,false,DType::Float64).double_data();
    size_t precise=0;
    for (double value:values) {
        check(value>=0. && value<1.,"Double uniform sample is outside unit support");
        precise+=value!=double(float(value));
    }
    check(precise>4000,"Double uniform has only Float32 precision");
    if (backend==Backend::CUDA) {
        const double next=std::nextafter(1.,2.);
        check(Tensor::uniform({4096},1.,next,19,backend,false,DType::Float64).double_data()==std::vector<double>(4096,1.),
              "CUDA Double uniform upper endpoint handling differs");
    }
}
void execution(Backend backend) {
    for (bool pooled:{false,true}) for (bool asynchronous:{false,true}) {
        AllocationPool pool(backend,0,pooled);
        ExecutionStream producer(backend),consumer(backend);
        for (int iteration=0;iteration<3;++iteration) {
            Tensor uniform,normal,output;
            {
                StreamScope scope(producer,asynchronous,true);
                uniform=Tensor::uniform({129},-2.,3.,17,backend,true,DType::Float64);
                normal=Tensor::normal({129},1.,2.,19,backend,true,DType::Float64);
                output=dropout(add(uniform,normal),.3,13);
            }
            {
                StreamScope scope(consumer,asynchronous,true);
                std::vector<double> mask=dropout(Tensor::ones({129},backend,false,DType::Float64),.3,13).double_data();
                std::vector<double> expected=add(uniform,normal).double_data();
                for (size_t i=0;i<expected.size();++i) expected[i]*=mask[i];
                near(output.double_data(),expected);
                output.backward(Tensor::ones({129},backend,false,DType::Float64));
                near(uniform.double_grad(),mask);near(normal.double_grad(),mask);
            }
        }
        producer.synchronize();consumer.synchronize();
        if (pooled) check(pool.stats().reuses>0,"Double random pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before=memory_stats(backend).live_bytes();
        raw_operations(backend);replay(backend);execution(backend);
        check(memory_stats(backend).live_bytes()==before,"Double random operation leaked storage");
        std::cout << "Float64 random: " << checks << " passed, 0 failed\n";
}

}  // namespace random

namespace raw_normalization {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double raw normalization operation was accepted");
}
void near(const std::vector<double>& values, const std::vector<double>& expected) {
    check(values.size() == expected.size(), "Double raw normalization result size differs");
    for (size_t i = 0; i < values.size(); ++i)
        check(std::abs(values[i] - expected[i]) < 2e-14, "Double raw normalization result differs");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
int bits(float value) { int result;std::memcpy(&result,&value,sizeof(value));return result; }

std::vector<int> attributes(const std::string& op, double epsilon=1e-5, bool wide=true) {
    std::vector<int> result=op=="batch_norm" ? std::vector<int>{bits(float(epsilon))} : std::vector<int>{1,bits(float(epsilon)),0};
    if (wide) {result.resize(8);std::memcpy(result.data()+5,&epsilon,sizeof(epsilon));result[7]=1;}
    return result;
}
void raw_operations(Backend backend) {
    for (const std::string op:{"layer_norm","rms_norm","batch_norm","group_norm","group_norm_input_grad","group_norm_weight_grad"}) {
        const bool dw=op=="group_norm_weight_grad",dx=op=="group_norm_input_grad",layer=op=="layer_norm" || op=="rms_norm";
        Tensor x=Tensor::from_double_data({2,3,5},{.1,.2,.3,.4,.5,.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4,1.5,
            -.1,-.2,-.3,-.4,-.5,-.6,-.7,-.8,-.9,-1.,-1.1,-1.2,-1.3,-1.4,-1.5},backend);
        Tensor weight=Tensor::ones({layer ? 5 : 3},backend,false,DType::Float64);
        Tensor bias=Tensor::ones(weight.shape(),backend,false,DType::Float64);
        Tensor gradient=mul(x,1.3);
        std::vector<Tensor> tensors=dw ? std::vector<Tensor>{x,gradient} : std::vector<Tensor>{x,weight,dx ? gradient : bias};
        Tensor output=Tensor::zeros(dw ? weight.shape() : x.shape(),backend,false,DType::Float64);
        const std::vector<int> attrs=attributes(op);
        KernelParams valid=descriptor(output,tensors);
        for (size_t i=0;i<attrs.size();++i) valid.v[KernelParams::kAttributeSlot+i]=attrs[i];
        std::vector<BufferPtr> inputs;
        for (const elman::Tensor& t:tensors) inputs.push_back(t.node()->value);
        backend_dispatch(op,output.node()->value,inputs,valid);
        const std::vector<uint8_t> before=output.raw_data();
        std::vector<std::pair<int,int>> changes{{KernelParams::kOutputCountSlot,0},{KernelParams::kOutputRankSlot,-1},
            {KernelParams::kOutputRankSlot,9},{KernelParams::kOutputRankSlot,0},{KernelParams::kOutputShapeSlot,-1},
            {KernelParams::kOutputShapeSlot,4},{KernelParams::kOutputDTypeSlot,int(DType::Float32)},
            {KernelParams::kAttributeSlot+7,2},{KernelParams::input_slot(KernelInput::D),0}};
        for (size_t i=0;i<tensors.size();++i) {
            const int slot=KernelParams::input_slot(KernelInput(i));
            changes.insert(changes.end(),{{slot,-1},{slot,9},{slot,0},{slot+KernelParams::kInputShapeOffset,-1},
                {slot+KernelParams::kInputShapeOffset,4},{slot+KernelParams::kInputStrideOffset,0},
                {KernelParams::input_dtype_slot(KernelInput(i)),int(DType::Float32)}});
        }
        if (op!="batch_norm") changes.push_back({KernelParams::kAttributeSlot,0});
        if (op=="group_norm" || dx || dw || op=="rms_norm") changes.push_back({KernelParams::kAttributeSlot+2,2});
        for (std::pair<int, int> change:changes) {
            KernelParams p=valid;p.v[change.first]=change.second;
            rejects([&] {backend_dispatch(op,output.node()->value,inputs,p);});
            check(output.raw_data()==before,"Rejected normalization descriptor changed output");
        }
        for (size_t i=0;i<tensors.size();++i) {
            std::vector<Tensor> mixed=tensors;mixed[i]=cast(mixed[i],DType::Float32);
            rejects([&]{dispatch_tensor_into(op,output,mixed,attrs);});
            if (backend==Backend::CUDA) {
                mixed=tensors;mixed[i]=mixed[i].to(Backend::CPU);
                rejects([&]{dispatch_tensor_into(op,output,mixed,attrs);});
            }
        }
        std::vector<BufferPtr> extra=inputs;extra.resize(4);extra[3]=x.node()->value;
        rejects([&]{backend_dispatch(op,output.node()->value,extra,valid);});
        rejects([&]{backend_dispatch(op,output.node()->value,{},valid);});
        rejects([&]{dispatch_tensor_into(op,output,tensors,attributes(op,-1.));});
        rejects([&]{dispatch_tensor_into(op,output,tensors,attributes(op,NAN));});
        Tensor root=Tensor::ones({61},backend,false,DType::Float64);
        Tensor first=contiguous_view(root,x.shape());first.assign_double(x.double_data());
        std::vector<Tensor> operands=tensors;operands[0]=first;
        const std::vector<uint8_t> original=root.raw_data();
        rejects([&]{dispatch_tensor_into(op,contiguous_view(root,output.shape()),operands,attrs);});
        rejects([&]{dispatch_tensor_into(op,contiguous_view(root,output.shape(),1),operands,attrs);});
        check(root.raw_data()==original,"Rejected normalization overlap changed storage");
        Tensor disjoint=contiguous_view(root,output.shape(),30);
        dispatch_tensor_into(op,disjoint,operands,attrs);
        check(disjoint.raw_data()==before,"Disjoint normalization views differ");
        if (!dx && !dw && op!="batch_norm") {
            Tensor w=Tensor::ones(weight.shape(),backend,false,DType::Float64);
            Tensor z=Tensor::zeros(weight.shape(),backend,false,DType::Float64);
            Tensor bare=dispatch_tensor(op,x.shape(),{x},attrs,DType::Float64);
            near(bare.double_data(),dispatch_tensor(op,x.shape(),{x,w,z},attrs,DType::Float64).double_data());
            Tensor bias_only=dispatch_tensor(op,x.shape(),{x,Tensor{},bias},attrs,DType::Float64);
            near(bias_only.double_data(),dispatch_tensor(op,x.shape(),{x,w,bias},attrs,DType::Float64).double_data());
        }
        std::vector<BufferPtr> too_small=inputs;too_small.pop_back();
        if (dx || dw || op=="batch_norm") rejects([&]{backend_dispatch(op,output.node()->value,too_small,valid);});
        // A zero dimension must not conceal an overflowing prefix or suffix.
        Tensor empty=Tensor::zeros({0},backend,false,DType::Float64);
        KernelParams overflow=descriptor(empty,{empty});
        const int slot=KernelParams::input_slot(KernelInput::A);
        overflow.v[slot]=3;overflow.v[KernelParams::kOutputRankSlot]=3;
        for (int d=0;d<3;++d) {
            int size=d==2 ? 0 : 50000;
            overflow.v[slot+KernelParams::kInputShapeOffset+d]=size;
            overflow.v[slot+KernelParams::kInputStrideOffset+d]=d==2 ? 1 : 0;
            overflow.v[KernelParams::kOutputShapeSlot+d]=size;
        }
        rejects([&]{backend_dispatch(op,empty.node()->value,{empty.node()->value},overflow);});
    }
}
void precision(Backend backend) {
    Tensor x=Tensor::from_double_data({2,2},{3.,4.,3.,4.},backend);
    Tensor w=Tensor::ones({2},backend,false,DType::Float64), b=Tensor::zeros({2},backend,false,DType::Float64);
    const double epsilon=1.+std::ldexp(1.,-40);
    for (const std::string op:{"layer_norm","rms_norm","group_norm"}) {
        Tensor output=dispatch_tensor(op,x.shape(),{x,w,b},attributes(op,epsilon),DType::Float64);
        const double inverse=1./std::sqrt((op=="rms_norm" ? 12.5 : .25)+epsilon);
        near(output.double_data(),op=="rms_norm" ? std::vector<double>{3.*inverse,4.*inverse,3.*inverse,4.*inverse} :
            std::vector<double>{-.5*inverse,.5*inverse,-.5*inverse,.5*inverse});
        check(output.raw_data()!=dispatch_tensor(op,x.shape(),{x,w,b},attributes(op,epsilon,false),DType::Float64).raw_data(),"Normalization narrowed Double epsilon");
    }
    near(dispatch_tensor("batch_norm",x.shape(),{x,b,w},attributes("batch_norm",epsilon),DType::Float64).double_data(),
        {3./std::sqrt(1.+epsilon),4./std::sqrt(1.+epsilon),3./std::sqrt(1.+epsilon),4./std::sqrt(1.+epsilon)});
}
void execution(Backend backend) {
    for (bool pooled:{false,true}) for (bool asynchronous:{false,true}) {
        AllocationPool pool(backend,0,pooled);
        ExecutionStream producer(backend),consumer(backend);
        for (int iteration=0;iteration<3;++iteration) {
            Tensor x,w,b,output;
            {
                StreamScope scope(producer,asynchronous,true);
                x=Tensor::normal({2,3,7},0.,1.,13,backend,false,DType::Float64);
                w=Tensor::normal({3},0.,1.,19,backend,false,DType::Float64);
                b=Tensor::normal({3},0.,1.,23,backend,false,DType::Float64);
                output=dispatch_tensor("group_norm",x.shape(),{x,w,b},attributes("group_norm"),DType::Float64);
            }
            {
                StreamScope scope(consumer,asynchronous,true);
                near(output.double_data(),group_norm(x,1,w,b,1e-5).double_data());
            }
        }
        producer.synchronize();consumer.synchronize();
        if (pooled) check(pool.stats().reuses>0,"Normalization pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before=memory_stats(backend).live_bytes();
        raw_operations(backend);precision(backend);execution(backend);
        check(memory_stats(backend).live_bytes()==before,"Double normalization leaked storage");
        std::cout << "Float64 raw normalization: " << checks << " passed, 0 failed\n";
}

}  // namespace raw_normalization

namespace reduction {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Float64 reduction was accepted");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    p.v[KernelParams::kAttributeSlot] = 1;
    return p;
}
void malformed(Backend backend) {
    Tensor input = Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend);
    Tensor output = Tensor::zeros({2}, backend, false, DType::Float64);
    const KernelParams valid = descriptor(output, {input});
    const std::vector<BufferPtr> inputs{input.node()->value};
    backend_dispatch("sum", output.node()->value, inputs, valid);
    check(output.double_data() == std::vector<double>{6, 15}, "Raw Double sum differs");
    const std::vector<uint8_t> before = output.raw_data();
    const int slot = KernelParams::input_slot(KernelInput::A);
    for (std::pair<int, int> bad : std::vector<std::pair<int, int>>{
            {slot, -1}, {slot, 9}, {slot + KernelParams::kInputShapeOffset, -1},
            {slot + KernelParams::kInputShapeOffset, 3}, {slot + KernelParams::kInputStrideOffset, 1},
            {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, -1},
            {KernelParams::kOutputRankSlot, 9}, {KernelParams::kOutputShapeSlot, 3},
            {KernelParams::kOutputDTypeSlot, int(DType::Float32)}, {KernelParams::kInputDTypeSlot, int(DType::Float32)},
            {KernelParams::kAttributeSlot, -2}, {KernelParams::kAttributeSlot, 2}, {KernelParams::kAttributeSlot + 1, 2}}) {
        KernelParams p = valid; p.v[bad.first] = bad.second;
        rejects([&] { backend_dispatch("sum", output.node()->value, inputs, p); });
        check(output.raw_data() == before, "Rejected Double reduction changed output");
    }
    rejects([&] { backend_dispatch("sum", output.node()->value, {}, valid); });
    rejects([&] { backend_dispatch("sum", output.node()->value, {{}}, valid); });
    rejects([&] { backend_dispatch("sum", output.node()->value, {inputs[0], inputs[0]}, valid); });
    Tensor narrow = cast(output, DType::Float32);
    rejects([&] { backend_dispatch("sum", narrow.node()->value, inputs, descriptor(narrow, {input})); });
    Tensor gradient = Tensor::zeros({2, 3}, backend, false, DType::Float64);
    Tensor seed = Tensor::ones({2}, backend, false, DType::Float64);
    KernelParams p = descriptor(gradient, {input, output, seed});
    for (int kind : {-1, 7}) {
        p.v[KernelParams::kAttributeSlot + 2] = kind;
        rejects([&] { backend_dispatch("reduce_grad", gradient.node()->value,
            {inputs[0], output.node()->value, seed.node()->value}, p); });
    }
    p = descriptor(gradient, {input, output, seed});
    p.v[KernelParams::kAttributeSlot + 1] = 1;
    rejects([&] { backend_dispatch("reduce_grad", gradient.node()->value,
        {inputs[0], output.node()->value, seed.node()->value}, p); });
    Tensor empty = Tensor::from_double_data({0}, {}, backend);
    rejects([&] { max(empty, 0); });
    check(sum_all(empty).double_data() == std::vector<double>{0}, "Empty Double sum differs");
    check(prod(empty, 0).double_data() == std::vector<double>{1}, "Empty Double product differs");
    check(std::isnan(mean_all(empty).double_data()[0]), "Empty Double mean differs");
}
void aliases(Backend backend) {
    Tensor root = Tensor::from_double_data({7}, {1, 2, 3, 4, 5, 6, 0}, backend);
    Tensor input = contiguous_view(root, {2, 3});
    Tensor output = contiguous_view(root, {2});
    const std::vector<uint8_t> before = root.raw_data();
    rejects([&] { dispatch_tensor_into("sum", output, {input}, {1, 0}); });
    rejects([&] { dispatch_tensor_into("cumsum", contiguous_view(root, {2, 3}, 1), {input}, {1}); });
    check(root.raw_data() == before, "Rejected Double reduction alias changed storage");
    dispatch_tensor_into("cumsum", input, {input}, {1});
    check(input.double_data() == std::vector<double>{1, 3, 6, 4, 9, 15}, "In-place Double scan differs");
    input.assign_double({1, 2, 3, 4, 5, 6});
    dispatch_tensor_into("cumsum_backward", input, {input}, {1});
    check(input.double_data() == std::vector<double>{6, 5, 3, 15, 11, 6}, "In-place reverse Double scan differs");
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 4; ++iteration) {
            Tensor value, input;
            {
                StreamScope scope(producer, asynchronous, true);
                input = Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend, true);
                value = cumsum(input, 1);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                Tensor result = sum_all(value);
                check(result.double_data() == std::vector<double>{38}, "Double reduction stream dependencies differ");
                result.backward();
                check(input.double_grad() == std::vector<double>{3, 2, 1, 3, 2, 1}, "Double scan backward lost its producer");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double reduction pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        malformed(backend); aliases(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Double reductions leaked storage");
        std::cout << "Float64 reduction: " << checks << " passed, 0 failed\n";
}

}  // namespace reduction

namespace sampling {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double sampling operation was accepted");
}
void near(const std::vector<double>& values, const std::vector<double>& expected) {
    check(values.size() == expected.size(), "Double sampling result size differs");
    for (size_t i = 0; i < values.size(); ++i)
        check(std::abs(values[i] - expected[i]) < 2e-14, "Double sampling result differs");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}

void raw_operations(Backend backend) {
    Tensor weights=Tensor::from_double_data({2,3},{.25,0.,.75,0.,.5,.5},backend);
    Tensor cumulative=cumsum(weights,1);
    Tensor uniform=Tensor::from_double_data({2},{.25,.5},backend);
    Tensor output=Tensor::zeros({2},backend,false,DType::Int32);
    std::vector<Tensor> tensors{weights,cumulative,uniform};
    std::vector<BufferPtr> inputs{weights.node()->value,cumulative.node()->value,uniform.node()->value};
    KernelParams valid=descriptor(output,tensors);
    backend_dispatch("categorical",output.node()->value,inputs,valid);
    check(output.int_data()==std::vector<int32_t>({2,2}),"Raw Double categorical boundary differs");
    const std::vector<uint8_t> before=output.raw_data();
    std::vector<std::pair<int,int>> changes{{KernelParams::kOutputCountSlot,0},{KernelParams::kOutputRankSlot,0},
        {KernelParams::kOutputRankSlot,-1},{KernelParams::kOutputRankSlot,9},{KernelParams::kOutputShapeSlot,3},
        {KernelParams::kOutputDTypeSlot,int(DType::Float64)},{KernelParams::input_slot(KernelInput::D),0}};
    for (int i=0;i<3;++i) {
        int slot=KernelParams::input_slot(KernelInput(i));
        changes.insert(changes.end(),{{slot,-1},{slot,9},{slot,0},{slot+KernelParams::kInputShapeOffset,3},
            {slot+KernelParams::kInputShapeOffset,-1},{slot+KernelParams::kInputStrideOffset,0},
            {KernelParams::input_dtype_slot(KernelInput(i)),int(DType::Float32)}});
        if (i<2) changes.insert(changes.end(),{{slot,1},{slot+KernelParams::kInputShapeOffset+1,0},
            {slot+KernelParams::kInputShapeOffset+1,INT32_MAX},{slot+KernelParams::kInputStrideOffset+1,2}});
        else changes.push_back({slot,2});
    }
    for (std::pair<int, int> change:changes) {
        KernelParams p=valid;p.v[change.first]=change.second;
        rejects([&]{backend_dispatch("categorical",output.node()->value,inputs,p);});
        check(output.raw_data()==before,"Rejected Double categorical descriptor changed output");
    }
    rejects([&]{backend_dispatch("categorical",output.node()->value,{},valid);});
    std::vector<BufferPtr> extra=inputs;extra.push_back(weights.node()->value);
    rejects([&]{backend_dispatch("categorical",output.node()->value,extra,valid);});
    for (int i=0;i<3;++i) {
        std::vector<Tensor> invalid=tensors;invalid[i]=cast(invalid[i],DType::Float32);
        rejects([&]{dispatch_tensor_into("categorical",output,invalid);});
        if (backend==Backend::CUDA) {
            invalid=tensors;invalid[i]=invalid[i].to(Backend::CPU);
            rejects([&]{dispatch_tensor_into("categorical",output,invalid);});
        }
        // Typed aliases are constructed only to exercise raw overlap validation.
        BufferPtr alias=backend_view(inputs[i],0,2);alias->dtype=DType::Int32;alias->bytes=8;
        const std::vector<uint8_t> original=tensors[i].raw_data();
        rejects([&]{backend_dispatch("categorical",alias,inputs,valid);});
        check(tensors[i].raw_data()==original,"Rejected categorical alias changed input storage");
    }
    Tensor root=Tensor::zeros({10},backend,false,DType::Float64);
    Tensor view=contiguous_view(root,{2,3});view.assign_double(weights.double_data());
    BufferPtr disjoint=backend_view(root.node()->value,6,2);disjoint->dtype=DType::Int32;disjoint->bytes=8;
    backend_dispatch("categorical",disjoint,{view.node()->value,inputs[1],inputs[2]},valid);
    check(backend_download_int(disjoint)==output.int_data(),"Disjoint categorical storage differs");
    Tensor empty=Tensor::zeros({0,3},backend,false,DType::Float64);
    check(dispatch_tensor("categorical",{0},{empty,empty,Tensor::zeros({0},backend,false,DType::Float64)},{},DType::Int32).numel()==0,
        "Empty Double categorical output differs");
    for (double value:{double(NAN),double(INFINITY),-double(INFINITY),-1.}) {
        Tensor invalid=weights;invalid=Tensor::from_double_data({2,3},{.25,value,.75,0.,.5,.5},backend);
        std::vector<int32_t> actual=dispatch_tensor("categorical",{2},{invalid,cumulative,uniform},{},DType::Int32).int_data();
        check(actual==std::vector<int32_t>({-1,2}),"Invalid Double categorical row was not isolated");
        rejects([&]{categorical_sample_double(invalid,{.25,.5});});
        Tensor invalid_cdf=Tensor::from_double_data({2,3},{.25,value,1.,0.,.5,1.},backend);
        check(dispatch_tensor("categorical",{2},{weights,invalid_cdf,uniform},{},DType::Int32).int_data()==std::vector<int32_t>({-1,2}),
            "Invalid Double cumulative row was accepted");
    }
    rejects([&]{categorical_sample_double(weights,{});});
    rejects([&]{categorical_sample_double(weights,{0.,1.});});
    rejects([&]{categorical_sample_double(cast(weights,DType::Float32),{.25,.5});});
    rejects([&]{categorical_sample_double(Tensor::zeros({0,3},backend,false,DType::Float64),{});});
}
void precision(Backend backend) {
    const double delta=std::ldexp(1.,-40);
    Tensor weights=Tensor::from_double_data({1,2},{.5+delta,.5-delta},backend);
    check(categorical_sample_double(weights,{.5+delta*.5})==std::vector<int32_t>{0},"Double CDF lost its sub-Float32 boundary");
    check(categorical_sample_double(weights,{.5+delta})==std::vector<int32_t>{1},"Double CDF equality was not strict");
    check(categorical_sample(weights,std::vector<float>{.5f})==std::vector<int32_t>{0},"Float uniform bridge narrowed Double weights");
    Tensor tiny=Tensor::from_double_data({1,2},{1e-300,2e-300},backend);
    check(categorical_sample_double(tiny,{.1})==std::vector<int32_t>{0},"Double categorical lost sub-Float32 weights");
    Tensor x=Tensor::from_double_data({1,2},{-2.,2.},backend,true);
    Tensor history=Tensor::from_int_data({1,3},{0,1,1},backend);
    SamplingOptions options;options.temperature=1.+delta*.25;options.repetition_penalty=1.+delta*.5;
    Tensor scores=sampling_scores(x,history,options);
    near(scores.double_data(),{-2.*options.repetition_penalty/options.temperature,2./options.repetition_penalty/options.temperature});
    check(scores.dtype()==DType::Float64 && !scores.requires_grad(),"Double sampling lost dtype or retained autograd");
    Tensor ties=Tensor::zeros({1,2},backend,false,DType::Float64);
    Tensor no_history=Tensor::from_int_data({1,0},{},backend);
    options={};options.top_p=.5+delta;
    check(sampling_scores(ties,no_history,options).double_data()==std::vector<double>({0.,0.}),"Double top-p cutoff was narrowed");
    options.top_p=.5-delta;std::vector<double> values=sampling_scores(ties,no_history,options).double_data();
    check(values[0]==-INFINITY && values[1]==0.,"Double top-p boundary differs");
    options={};options.temperature=1e100;
    values=sampling_scores(x,no_history,options).double_data();
    check(std::abs(values[0]/(-2e-100)-1.)<1e-14 && std::abs(values[1]/2e-100-1.)<1e-14,"Double temperature lost tiny scores");
    options.temperature=1e-100;values=sampling_scores(x,no_history,options).double_data();
    check(std::isfinite(values[0]) && values[0]<-1e99 && values[1]>1e99,"Double temperature was narrowed");
    for (uint64_t seed:{uint64_t(0),uint64_t(1),uint64_t(42),uint64_t(0xffffffffffffffffull)}) {
        SamplingRng wide(seed),legacy(seed),replay(seed);
        int narrow=0;
        for (int i=0;i<4097;++i) {
            const double value=wide.next_double();
            check(value>=0. && value<1. && value==replay.next_double(),"Double sampling RNG range or replay differs");
            check(float(std::floor(value*16777216.)/16777216.)==legacy.next(),"Double sampling RNG changed the legacy stream");
            narrow+=value==double(float(value));
        }
        check(narrow<41,"Double sampling RNG has only Float32 precision");
        check(wide.next()==legacy.next(),"Double sampling RNG advanced an unexpected number of words");
    }
}
void execution(Backend backend) {
    for (bool pooled:{false,true}) for (bool asynchronous:{false,true}) {
        AllocationPool pool(backend,0,pooled);ExecutionStream producer(backend),consumer(backend);
        for (int iteration=0;iteration<3;++iteration) {
            Tensor x,history,scores;std::vector<int32_t> first,second;
            SamplingOptions options;options.top_k=5;options.top_p=.7;options.repetition_penalty=1.3;options.temperature=.6;
            {
                StreamScope scope(producer,asynchronous,true);
                x=Tensor::normal({3,17},0.,1.,13,backend,false,DType::Float64);
                history=Tensor::from_int_data({3,3},{0,2,2,1,2,7,6,6,3},backend);
                scores=sampling_scores(x,history,options);
                SamplingRng rng(42);first=sample_tokens(x,history,options,rng);
            }
            {
                StreamScope scope(consumer,asynchronous,true);
                SamplingRng rng(42);second=sample_tokens(x,history,options,rng);
                check(first==second,"Cross-stream Double sampling replay differs");
                std::vector<double> draws;SamplingRng explicit_rng(42);for (int i=0;i<3;++i) draws.push_back(explicit_rng.next_double());
                check(first==categorical_sample_double(softmax(scores,1),draws),"Seeded Double sampling differs from explicit uniforms");
            }
        }
        producer.synchronize();consumer.synchronize();
        if (pooled) check(pool.stats().reuses>0,"Double sampling pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before=memory_stats(backend).live_bytes();
        raw_operations(backend);precision(backend);execution(backend);
        check(memory_stats(backend).live_bytes()==before,"Double sampling leaked storage");
        std::cout << "Float64 sampling: " << checks << " passed, 0 failed\n";
}

}  // namespace sampling

namespace softmax_tests {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double softmax operation was accepted");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}

void malformed(Backend backend) {
    Tensor input = Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend);
    Tensor output = Tensor::zeros({2, 3}, backend, false, DType::Float64);
    KernelParams valid = descriptor(output, {input});
    valid.v[KernelParams::kAttributeSlot] = 1;
    const std::vector<BufferPtr> inputs{input.node()->value};
    backend_dispatch("softmax", output.node()->value, inputs, valid);
    check(output.double_data() == softmax(input, 1).double_data(), "Raw Double softmax differs");
    const std::vector<uint8_t> before = output.raw_data();
    const int slot = KernelParams::input_slot(KernelInput::A);
    for (std::pair<int, int> change : std::vector<std::pair<int, int>>{
            {slot, -1}, {slot, 9}, {slot + KernelParams::kInputShapeOffset, -1},
            {slot + KernelParams::kInputShapeOffset, 3}, {slot + KernelParams::kInputStrideOffset, 1},
            {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, -1},
            {KernelParams::kOutputRankSlot, 9}, {KernelParams::kOutputShapeSlot, 3},
            {KernelParams::kOutputDTypeSlot, int(DType::Float32)}, {KernelParams::kInputDTypeSlot, int(DType::Float32)},
            {KernelParams::kAttributeSlot, -1}, {KernelParams::kAttributeSlot, 2}}) {
        KernelParams p = valid; p.v[change.first] = change.second;
        rejects([&] { backend_dispatch("softmax", output.node()->value, inputs, p); });
        check(output.raw_data() == before, "Rejected Double softmax changed output");
    }
    rejects([&] { backend_dispatch("softmax", output.node()->value, {}, valid); });
    rejects([&] { backend_dispatch("softmax", output.node()->value, {inputs[0], inputs[0]}, valid); });
    Tensor narrow = cast(output, DType::Float32);
    rejects([&] { backend_dispatch("softmax", narrow.node()->value, inputs, descriptor(narrow, {input})); });
    if (backend == Backend::CUDA) {
        Tensor other = Tensor::ones({2, 3}, Backend::CPU, false, DType::Float64);
        rejects([&] { dispatch_tensor_into("softmax", output, {other}, {1}); });
    }
}
void aliases(Backend backend) {
    Tensor root = Tensor::from_double_data({7}, {1, 2, 3, 4, 5, 6, 0}, backend);
    Tensor input = contiguous_view(root, {2, 3});
    const std::vector<uint8_t> before = root.raw_data();
    rejects([&] { dispatch_tensor_into("softmax", contiguous_view(root, {2, 3}, 1), {input}, {1}); });
    check(root.raw_data() == before, "Rejected Double softmax alias changed storage");
    for (const char* operation : {"softmax", "log_softmax"}) for (int axis : {0, 1}) {
        input.assign_double({1, 2, 3, 4, 5, 6});
        std::vector<double> expected = dispatch_tensor(operation, input.shape(), {input}, {axis}, DType::Float64).double_data();
        dispatch_tensor_into(operation, input, {input}, {axis});
        check(input.double_data() == expected, "In-place Double softmax differs");
    }
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 4; ++iteration) {
            Tensor value, input;
            {
                StreamScope scope(producer, asynchronous, true);
                input = Tensor::zeros({2, 3}, backend, true, DType::Float64);
                value = softmax(input, 1);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                check(value.double_data() == std::vector<double>(6, 1. / 3.), "Double softmax stream dependency differs");
                value.backward(Tensor::from_double_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend));
                std::vector<double> gradient = input.double_grad();
                for (size_t i = 0; i < gradient.size(); ++i)
                    check(std::abs(gradient[i] - (int(i % 3) - 1) / 3.) < 2e-15, "Double softmax backward differs");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double softmax pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        malformed(backend); aliases(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Double softmax leaked storage");
        std::cout << "Float64 softmax: " << checks << " passed, 0 failed\n";
}

}  // namespace softmax_tests

namespace spatial {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Double spatial operation was accepted");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs, const std::vector<int>& attrs) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t i = 0; i < attrs.size(); ++i) p.v[KernelParams::kAttributeSlot + i] = attrs[i];
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void raw_operations(Backend backend) {
    Tensor x = Tensor::from_double_data({1,1,3,3}, {1,2,3,4,5,6,7,8,9}, backend);
    Tensor w = Tensor::ones({1,1,2,2}, backend, false, DType::Float64);
    Tensor bias = Tensor::zeros({1}, backend, false, DType::Float64);
    Tensor g = Tensor::ones({1,1,2,2}, backend, false, DType::Float64);
    const std::vector<int> convolution{1,1,0,0,1,1,1}, pooling{2,2,1,1,0,0,1};
    struct Case { const char* name; Shape shape; std::vector<Tensor> inputs; std::vector<int> attrs; std::vector<double> expected; };
    const std::vector<Case> cases{
        {"conv2d", {1,1,2,2}, {x,w,bias}, convolution, {12,16,24,28}},
        {"conv2d", {1,1,2,2}, {x,w}, convolution, {12,16,24,28}},
        {"conv2d_input_grad", x.shape(), {g,w}, convolution, {1,2,1,2,4,2,1,2,1}},
        {"conv2d_weight_grad", w.shape(), {x,g}, convolution, {12,16,24,28}},
        {"max_pool2d", {1,1,2,2}, {x}, pooling, {5,6,8,9}},
        {"avg_pool2d", {1,1,2,2}, {x}, pooling, {3,4,6,7}},
        {"max_pool2d_grad", x.shape(), {x,g}, pooling, {0,0,0,0,1,1,0,1,1}},
        {"avg_pool2d_grad", x.shape(), {x,g}, pooling, {.25,.5,.25,.5,1,.5,.25,.5,.25}}
    };
    for (const Case& test : cases) {
        Tensor output = Tensor::zeros(test.shape, backend, false, DType::Float64);
        KernelParams valid = descriptor(output, test.inputs, test.attrs);
        std::vector<BufferPtr> inputs;
        for (const elman::Tensor& input : test.inputs) inputs.push_back(input.node()->value);
        backend_dispatch(test.name, output.node()->value, inputs, valid);
        check(output.double_data() == test.expected, "Raw Double spatial result differs");
        const std::vector<uint8_t> before = output.raw_data();
        const int slot = KernelParams::input_slot(KernelInput::A);
        for (std::pair<int, int> change : std::vector<std::pair<int,int>>{
                {slot,-1},{slot,9},{slot,3},{slot+KernelParams::kInputShapeOffset,-1},
                {slot+KernelParams::kInputShapeOffset,2},{slot+KernelParams::kInputStrideOffset,1},
                {KernelParams::kOutputCountSlot,1},{KernelParams::kOutputRankSlot,-1},{KernelParams::kOutputRankSlot,9},
                {KernelParams::kOutputShapeSlot,2},{KernelParams::kInputDTypeSlot,int(DType::Float32)},
                {KernelParams::kOutputDTypeSlot,int(DType::Float32)},
                {KernelParams::input_slot(KernelInput::D),0}}) {
            KernelParams p = valid; p.v[change.first] = change.second;
            rejects([&] { backend_dispatch(test.name, output.node()->value, inputs, p); });
            check(output.raw_data() == before, "Rejected Double spatial descriptor changed output");
        }
        const bool conv = std::string(test.name).find("conv2d") == 0;
        std::vector<std::pair<int, int>> bad_attributes = conv ? std::vector<std::pair<int,int>>{{0,0},{1,0},{2,-1},{3,-1},{4,0},{5,0},{6,0},{6,2}} :
                                    std::vector<std::pair<int,int>>{{0,0},{1,0},{2,0},{3,0},{4,-1},{5,-1},{4,2},{6,2}};
        for (std::pair<int, int> change : bad_attributes) {
            KernelParams p = valid; p.v[KernelParams::kAttributeSlot + change.first] = change.second;
            rejects([&] { backend_dispatch(test.name, output.node()->value, inputs, p); });
            check(output.raw_data() == before, "Rejected Double spatial attributes changed output");
        }
        rejects([&] { backend_dispatch(test.name, output.node()->value, {}, valid); });
        std::vector<BufferPtr> extra = inputs; extra.push_back(x.node()->value);
        rejects([&] { backend_dispatch(test.name, output.node()->value, extra, valid); });
        Tensor root = Tensor::zeros({int(output.numel()+test.inputs[0].numel())}, backend, false, DType::Float64);
        std::vector<Tensor> operands = test.inputs;
        operands[0] = contiguous_view(root, test.inputs[0].shape());
        rejects([&] { dispatch_tensor_into(test.name, contiguous_view(root,test.shape), operands, test.attrs); });
        operands[0].assign_double(test.inputs[0].double_data());
        Tensor disjoint = contiguous_view(root,test.shape,test.inputs[0].numel());
        dispatch_tensor_into(test.name, disjoint, operands, test.attrs);
        check(disjoint.double_data() == test.expected, "Disjoint Double spatial storage views differ");
        if (backend == Backend::CUDA) {
            operands = test.inputs; operands[0] = operands[0].to(Backend::CPU);
            rejects([&] { dispatch_tensor_into(test.name,output,operands,test.attrs); });
        }
    }
    Tensor wrong_bias = Tensor::ones({2},backend,false,DType::Float64);
    rejects([&] { conv2d(x,w,wrong_bias); });
    rejects([&] { conv2d(x,cast(w,DType::Float32),bias); });
    rejects([&] { max_pool2d(x,2,1,2); });
    rejects([&] { avg_pool2d(x,2,1,-1); });
    Tensor empty = Tensor::zeros({0,1,3,3},backend,false,DType::Float64);
    check(conv2d(empty,w,bias).numel() == 0 && max_pool2d(empty,2).numel() == 0, "Double spatial empty batch differs");
}
void execution(Backend backend) {
    for (bool pooled : {false,true}) for (bool asynchronous : {false,true}) {
        AllocationPool pool(backend,0,pooled);
        ExecutionStream producer(backend),consumer(backend);
        for (int iteration=0;iteration<3;++iteration) {
            Tensor input,weight,bias,output;
            {
                StreamScope scope(producer,asynchronous,true);
                input = Tensor::from_double_data({11},{77,1,2,3,4,5,6,7,8,9,88},backend,true);
                weight = Tensor::from_double_data({6},{33,1,1,1,1,44},backend,true);
                bias = Tensor::from_double_data({3},{55,0,66},backend,true);
                Tensor value = conv2d(contiguous_view(input,{1,1,3,3},1),contiguous_view(weight,{1,1,2,2},1),contiguous_view(bias,{1},1));
                output = max_pool2d(value,2,1);
            }
            {
                StreamScope scope(consumer,asynchronous,true);
                check(output.double_data() == std::vector<double>{28}, "Double spatial stream result differs");
                output.backward(Tensor::from_double_data({1,1,1,1},{3},backend));
                check(input.double_grad() == std::vector<double>{0,0,0,0,0,3,3,0,3,3,0}, "Double convolution view input gradient differs");
                check(weight.double_grad() == std::vector<double>{0,15,18,24,27,0}, "Double convolution view weight gradient differs");
                check(bias.double_grad() == std::vector<double>{0,3,0}, "Double convolution view bias gradient differs");
                check(input.double_data() == std::vector<double>{77,1,2,3,4,5,6,7,8,9,88} &&
                      weight.double_data() == std::vector<double>{33,1,1,1,1,44} && bias.double_data() == std::vector<double>{55,0,66},
                      "Double spatial operation mutated source storage");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Double spatial pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const size_t before = memory_stats(backend).live_bytes();
        raw_operations(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Double spatial operation leaked storage");
        std::cout << "Float64 spatial: " << checks << " passed, 0 failed\n";
}

}  // namespace spatial

}  // namespace

int main(int argc, char** argv) {
    try {
        const elman::Backend backend = elman::parse_backend(argc > 1 ? argv[1] : "cpu");
        if (!elman::backend_available(backend)) return 77;
        const std::filesystem::path output = argc > 2 ? argv[2] : "TestArtifacts/Float64";
        const std::chrono::steady_clock::rep stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::filesystem::path optimizer_output = output / std::to_string(stamp);
        attention::run(backend);
        compatibility::run(backend);
        fused::run(backend);
        indexing::run(backend);
        loss::run(backend);
        matrix::run(backend);
        normalization::run(backend);
        optimizer::run(backend, optimizer_output);
        pointwise::run(backend);
        random::run(backend);
        raw_normalization::run(backend);
        reduction::run(backend);
        sampling::run(backend);
        softmax_tests::run(backend);
        spatial::run(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
