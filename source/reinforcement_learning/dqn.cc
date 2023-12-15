/**
 * @file dqn.cc
 * @brief Source file for class dqn
 */

#include "dqn.h"
#include "domain.h"

#include <torch/torch.h>

using namespace std;


///////////////////////////////////////////////////////////////////////////////
/** dqn initializer
 *
 * @param p_domn  \input set domain pointer with.
 */

void dqn::init(domain *p_domn) {
    domn    = p_domn;
}


///////////////////////////////////////////////////////////////////////////////
/** dqn constructor
 */
dqn::dqn(int n_observations, int n_actions, int n_neurons_per_layer) {
    layer1 = register_module("layer1", torch::nn::Linear(n_observations, n_neurons_per_layer));
    layer2 = register_module("layer2", torch::nn::Linear(n_neurons_per_layer, n_neurons_per_layer));
    layer3 = register_module("layer3", torch::nn::Linear(n_neurons_per_layer, n_neurons_per_layer));
}


///////////////////////////////////////////////////////////////////////////////
/** Forward pass
 *     
 * Called with either one element to determine next action, or a batch
 * during optimization. 
 * 
 * @param x     \input tensor of input data
 * 
 * Returns:
 * tensor([[left0exp,right0exp]...]).
 */
torch::Tensor dqn::forward(torch::Tensor x) {
    x = torch::relu(layer1->forward(x));
    x = torch::relu(layer2->forward(x));
    x = layer3->forward(x);
    return x;
}


///////////////////////////////////////////////////////////////////////////////
/** dqn destructor
 */
dqn::~dqn() {
    delete layer1;
    delete layer2;
    delete layer3;
}
