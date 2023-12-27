#include <torch/torch.h>
#include <dqn.h>

int main() {
  
  // Create a new Net.
  auto net = std::make_shared<dqn>(2,2,5);
  torch::save(net, "net.pt");
  
}