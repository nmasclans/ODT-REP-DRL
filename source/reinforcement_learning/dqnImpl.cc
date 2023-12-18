/**
 * @file dqnImpl.cc
 * @brief Source file for class dqnImpl
 */

#include "dqnImpl.h"
#include "domain.h"

#include <torch/torch.h>

using namespace std;


///////////////////////////////////////////////////////////////////////////////
/**General information:
 * The Deep Q-Network (DQN) model is a feed forward neural netework that takes
 * in the difference between the current and the previous screen patches.
 * By 18-dec 2023, it has two possible discrete output values: Q(s, action=move_left) 
 * and Q(s,action=move_right), where state = s is the networks input.
 * In effect, the network is trying to predict the EXPECTED RETURN of taking
 * each action given the current input.  
 */

///////////////////////////////////////////////////////////////////////////////
/** dqn initializer
 *
 * @param p_domn  \input set domain pointer with.
 */

void dqnImpl::init(domain *p_domn) {
    domn    = p_domn;
}


///////////////////////////////////////////////////////////////////////////////
/** dqn constructor
 */
dqnImpl::dqnImpl(int n_observations, int n_actions, int n_neurons_per_layer)
    : layer1(torch::nn::Linear(n_observations, n_neurons_per_layer)),
      layer2(torch::nn::Linear(n_neurons_per_layer, n_neurons_per_layer)),
      layer3(torch::nn::Linear(n_neurons_per_layer, n_neurons_per_layer)) {
    
    // register_module() is needed if we want to use the parameters() method later on
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);

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
torch::Tensor dqnImpl::forward(torch::Tensor x) {
    x = torch::relu(layer1->forward(x));
    x = torch::relu(layer2->forward(x));
    x = layer3->forward(x);
    return x;
}


///////////////////////////////////////////////////////////////////////////////
/** dqn destructor
 */
dqnImpl::~dqnImpl() {
    delete layer1;
    delete layer2;
    delete layer3;
}


TORCH_MODULE(dqnImpl);