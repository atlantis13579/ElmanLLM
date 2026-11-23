#include "Inference/GraphSession.h"
#include "IO/Json.h"
#include "IO/NumPy.h"
#include "Training/GraphTraining.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void gradients(Backend backend) {
    const Json graph = Json::parse(R"({
      "inputs":{"indices":{"dtype":"int64","shape":[3]}},
      "outputs":{"loss":{"shape":[]}},
      "nodes":[
        {"op":"Gather","inputs":["x","indices"],"output":"selected","attributes":{"axis":1}},
        {"op":"Log","inputs":["selected"],"output":"log"},
        {"op":"CumSum","inputs":["log"],"output":"scan","attributes":{"axis":1}},
        {"op":"Exp","inputs":["scan"],"output":"products"},
        {"op":"SumAll","inputs":["products"],"output":"loss"}
      ]})");
    std::vector<float> data{0.2f,0.7f,0.6f,0.4f,0.8f,0.5f};
    Tensor parameter = Tensor::from_data({2,3}, data, backend, true);
    const auto session = make_graph_session(graph, {{"x",parameter}}, backend, backend == Backend::CPU ? -1 : 0);
    NamedTensors inputs{{"indices",Tensor::from_int64_data({3},{2,0,1},backend)}};
    check(!session.run(inputs).at("loss").requires_grad(), "Inference unexpectedly recorded gradients");
    session.run_with_grad(inputs).at("loss").backward();
    const auto gradient = parameter.grad();
    for (size_t i=0; i<data.size(); ++i) {
        auto perturbed=data;perturbed[i]+=0.001f;parameter.assign(perturbed);
        const float plus=session.run(inputs).at("loss").data()[0];
        perturbed[i]-=0.002f;parameter.assign(perturbed);
        const float minus=session.run(inputs).at("loss").data()[0];
        check(std::abs(gradient[i]-(plus-minus)/0.002f)<0.001f,"Scan/gather gradient differs from finite differences");
    }
}
void training(Backend backend,const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    Json config=Json::parse(R"({
      "format":"elman.training_graph","version":1,
      "tensors":{"w":{"shape":[1],"data":[0],"trainable":true},"target":{"shape":[1],"data":[2]}},
      "inputs":{"target":{"tensor":"target"}},
      "graph":{"inputs":{"target":{"value":"input","shape":[1]}},"outputs":{"loss":{"shape":[]}},
        "nodes":[{"op":"MSELoss","inputs":["w","input"],"output":"loss"}]},
      "training":{"steps":150,"learning_rate":0.08,"evaluation_batches":1,"log_every":150,"require_loss_decrease":true},
      "output":{"metrics":"metrics.json","tensors":[{"file":"weights.npy","values":{"w":"w"}}]}
    })");
    config.write((directory/"job.json").u8string());
    RunOptions options;options.backend=backend_name(backend);train_graph(directory/"job.json",options);
    const auto values=load_tensor_archive(directory/"weights.npy",Backend::CPU).at("input").data();
    check(std::abs(values[0]-2)<0.01f,"JSON optimizer did not fit the parameter");
    const auto metrics=Json::read(directory/"metrics.json");check(metrics.at("final_loss").number()<1e-4,"JSON evaluation loss is incorrect");
    for(auto dtype:{DType::Float32,DType::Float64,DType::Int32,DType::Int64}) {
        const auto input=cast(Tensor::from_data({2,2},{1,-2,3,4}),dtype);io::save_npy(directory/"roundtrip.npy",input);
        const auto output=load_tensor_archive(directory/"roundtrip.npy",Backend::CPU).at("input");
        check(output.shape()==input.shape() && output.dtype()==dtype && output.raw_data()==input.raw_data(),"NPY round trip changed data");
    }
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::invalid_argument("Expected backend and artifact directory");
        const auto backend=parse_backend(argv[1]);gradients(backend);training(backend,std::filesystem::u8path(argv[2]));
        std::cout<<"Graph gradients, JSON training, and NPY round trips passed\n";return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
