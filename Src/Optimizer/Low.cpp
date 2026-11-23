#include "Optimizer/OptimizerInternal.h"
#include "Core/LocalFunction.h"
#include "Core/TensorDispatch.h"
#include "Operators/low_precision_optimizer.h"
#include <cmath>
#include <cstring>

namespace elman::optimizer_internal {
namespace {
std::vector<int> step_attributes(const std::string& algorithm, const StepParameters& step,
                                 bool first_step, double correction2) {
    std::vector<int> attributes(optimizer_attr::kUsedAttributes);
    attributes[optimizer_attr::kAlgorithm]=algorithm=="sgd"?0:algorithm=="adam"?1:2;
    attributes[optimizer_attr::kFirstStep]=first_step?1:0;
    const double scalars[]={step.rate,step.gradient_scale,step.weight_decay,step.momentum,step.beta1,step.beta2,step.epsilon,
        step.correction1,correction2};
    for(int i=0;i<optimizer_attr::kScalarCount;++i)std::memcpy(attributes.data()+optimizer_attr::kScalars+2*i,&scalars[i],8);
    return attributes;
}
}
LowStep low_step(const std::string& algorithm,const Tensor& values,const Tensor& gradients,
        const Tensor& first_moment,const Tensor& second_moment,const StepParameters& step,bool first_step){
    std::vector<int> attributes = step_attributes(algorithm, step, first_step, std::pow(step.correction2, .5));
    const std::vector<Tensor> inputs{values,gradients,first_moment,second_moment};
    LocalFunction result{[&](int mode){attributes[optimizer_attr::kResult]=mode;
        return dispatch_tensor("optimizer_step",values.shape(),inputs,attributes,mode==0?values.dtype():DType::Float32);}};
    return {result(0),result(1),result(2),result(3)};
}

TensorStep resident_float32_step(const std::string& algorithm, const Tensor& values, const Tensor& gradients,
                                const Tensor& first_moment, const Tensor& second_moment, const StepParameters& step) {
    std::vector<int> attributes = step_attributes(algorithm, step, false, step.correction2);
    attributes[optimizer_attr::kResidentFloat32] = 1;
    const std::vector<Tensor> inputs{values, gradients, first_moment, second_moment};
    LocalFunction result{[&](int mode) {
        attributes[optimizer_attr::kResult] = mode;
        return dispatch_tensor("optimizer_step", values.shape(), inputs, attributes, DType::Float32);
    }};
    return {result(0), result(1), algorithm == "sgd" ? second_moment : result(2)};
}
} // namespace elman::optimizer_internal
