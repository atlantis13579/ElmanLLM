#include "Operators/low_precision_optimizer.h"
#include <cmath>
#include <stdexcept>

namespace elman {
void validate_low_precision_optimizer(const BufferPtr& output,const std::vector<BufferPtr>& inputs,const KernelParams& p){
    void (*const require)(bool, const char*)=[](bool valid,const char* message){if(!valid)throw std::invalid_argument(message);};
    require(inputs.size()==4&&inputs[0]&&inputs[1]&&inputs[2]&&inputs[3],"Optimizer step requires four operands");
    const bool resident_float32 = inputs[0]->backend == Backend::CUDA && inputs[0]->dtype == DType::Float32;
    require((is_low_precision_float(inputs[0]->dtype)||resident_float32)&&inputs[1]->dtype==inputs[0]->dtype&&
        inputs[2]->dtype==DType::Float32&&inputs[3]->dtype==DType::Float32,"Invalid optimizer step parameter, gradient or moment dtype");
    const int algorithm=p.attribute(optimizer_attr::kAlgorithm),mode=p.attribute(optimizer_attr::kResult);
    require(algorithm>=0&&algorithm<=2&&mode>=0&&mode<=3&&
        (p.attribute(optimizer_attr::kFirstStep)==0||p.attribute(optimizer_attr::kFirstStep)==1)&&
        p.attribute(optimizer_attr::kResidentFloat32)==int(resident_float32),"Invalid optimizer step mode");
    require(!resident_float32 || (mode <= 2 && p.attribute(optimizer_attr::kFirstStep) == 0), "Invalid Float32 optimizer step mode");
    require(output->dtype==(mode==0?inputs[0]->dtype:DType::Float32),"Invalid optimizer step output dtype");
    const int rank=p.output_rank();require(rank>=0&&rank<=8,"Invalid optimizer step rank");
    for(int i=0;i<4;++i)require(p.input_rank(KernelInput(i))==rank,"Optimizer step operand rank differs");
    size_t forward=1,reverse=1;
    for(int d=0;d<rank;++d){const int length=p.output_shape(d),backward=p.output_shape(rank-d-1);
        require(length>=0&&backward>=0&&(length==0||forward<=size_t(INT32_MAX)/size_t(length))&&
            (backward==0||reverse<=size_t(INT32_MAX)/size_t(backward)),"Optimizer step dimensions overflow");
        for(int i=0;i<4;++i)require(p.input_shape(KernelInput(i),d)==length&&
            p.input_stride(KernelInput(i),rank-d-1)==int(reverse),"Optimizer step requires matching contiguous operands");
        forward*=size_t(length);reverse*=size_t(backward);
    }
    require(output->count==forward&&p.output_count()==int(forward),"Optimizer step output count differs");
    for(int i=0;i<4;++i){require(inputs[i]->count==forward,"Optimizer step input count differs");
        if(output->bytes&&inputs[i]->bytes){const uintptr_t a=reinterpret_cast<uintptr_t>(output->raw_pointer()),b=reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
            require(!(a<=b?b-a<output->bytes:a-b<inputs[i]->bytes),"Overlapping optimizer step buffers");}}
    double values[optimizer_attr::kScalarCount];
    for(int i=0;i<optimizer_attr::kScalarCount;++i){values[i]=optimizer_parameter(p,i);require(std::isfinite(values[i]),"Nonfinite optimizer step scalar");}
    require(values[0]>=0.&&values[1]>=0.&&values[2]>=0.&&values[3]>=0.&&values[3]<1.&&
        values[4]>=0.&&values[4]<1.&&values[5]>=0.&&values[5]<1.&&values[6]>0.&&
        values[7]>0.&&values[7]<=1.&&values[8]>0.&&values[8]<=1.,"Invalid optimizer step scalar domain");
    for(int i=optimizer_attr::kUsedAttributes;i<KernelParams::kAttributeCount;++i)require(p.attribute(i)==0,"Unexpected optimizer step attribute");
}
} // namespace elman
