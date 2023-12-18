#include <torch/torch.h>
#include <iostream>

using namespace std;

struct Net : torch::nn::Module {
  // this would go in .cc file
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  // this would go in .h file
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

// (understanding shared ownership, shared pointers)
struct Net2Impl : torch::nn::Module {};
TORCH_MODULE(Net2);

// (inspect module ownership) functions that takes a network as input
void a(Net  net) {}
void b(Net &net) {}
void c(Net *net) {}
void d(Net2 net) {}


////////////////////////////////////////////////////////////////////////////
int main() {

  cout << "------ test torch ------" << endl;
  torch::Tensor tensor = torch::eye(3);
  cout << tensor << endl << endl;

  cout << "------ simple net: print parameters ------" << endl;
  Net net(4, 5);
  // print network parameters - only value (+type)
  for (const auto& p : net.parameters()) {
    cout << p << endl;
  }
  cout << endl;
  // print network parameters - value and name
  for (const auto& pair : net.named_parameters()){
    cout << pair.key() << ":" << pair.value() << endl << endl;
  } 

  cout << "------ run network in forward mode ------" << endl;
  cout << net.forward(torch::ones({2,4})) << endl << endl;

  cout << "------ module ownership ------" << endl;
  // function a: void a(Net net) {}
  a(net);
  a(std::move(net));
  // function b: void b(Net &net) {}
  b(net);
  // function c: void c(Net *net) {}
  c(&net);
  // TORCH_MODULE
  Net2 net2;
  d(net2);

  cout << "------ dataset - dataloader------" << endl;
  // instantiate dataset
  auto dataset = torch::data::datasets::MNIST("../mnist")
    .map(torch::data::transforms::Normalize<>(0.5,0.5)) // from range 0,1 to -1,1
    .map(torch::data::transforms::Stack<>());
  // create new dataloader (class DataLoader, but 'auto' detects it already)
  // auto data_loader  = torch::data::make_data_loader(std::move(dataset));
  // -> speeding up the creation of the dataloader:
  auto data_loader   = torch::data::make_data_loader(
   std::move(dataset),
   torch::data::DataLoaderOptions().batch_size(64).workers(2));
  // load batches of data:
  cout << "Print batch data:" << endl;
  for (torch::data::Example<>& batch : *data_loader) {
    cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); i++) {
        cout << batch.target[i].item<int64_t>() << " ";
    }
    cout << endl;
  }
  /**The type returned by the data loader in this case is a torch::data::Example.
   * This type is a simple struct with a data field for the data and a target
   * field for the label. Because we applied the Stack collation earlier, 
   * the data loader returns only a single such example. If we had not applied 
   * the collation, the data loader would yield std::vector<torch::data::Example<>>
   * instead, with one element per example in the batch.*/
  





}