/**
 * @file actorCritic.cc
 * @brief Source file for class \ref actorCritic
 */

#include "actorCritic.h"

#include <string>
#include <torch/torch.h>
#include <vector>
#include <math.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
/** Constructor of Actor-Critic network architecture
 *
 * @param input_size            \input size of input  tensor of Actor-Critic network (int) 
 * @param output_size           \input size of output tensor of Actor-Critic network (int)
 * @param num_neurons_per_layer \input
 * @param std                   \input
 */
actorCritic::actorCritic(int input_size, int output_size, int num_neurons_per_layer, double std) : 
    // Actor network
    a_lin1_(torch::nn::Linear(input_size,            num_neurons_per_layer)),
    a_lin2_(torch::nn::Linear(num_neurons_per_layer, num_neurons_per_layer)),
    a_lin3_(torch::nn::Linear(num_neurons_per_layer, output_size)),
    mu_(torch::full(output_size, 0.0)),         // mean tensor, output of action network
    log_std_(torch::full(output_size, std)),    // std tensor
    // Critic network
    c_lin1_(torch::nn::Linear(input_size,            num_neurons_per_layer)),
    c_lin2_(torch::nn::Linear(num_neurons_per_layer, num_neurons_per_layer)),
    c_lin3_(torch::nn::Linear(num_neurons_per_layer, output_size)),
    c_val_(torch::nn::Linear(output_size,            1))
{
    // Register the modules
    // Action network
    register_module("a_lin1",     a_lin1_);
    register_module("a_lin2",     a_lin2_);
    register_module("a_lin3",     a_lin3_);
    register_parameter("log_std", log_std_);
    // Critic network
    register_module("c_lin1",     c_lin1_);
    register_module("c_lin2",     c_lin2_);
    register_module("c_lin3",     c_lin3_);
    register_module("c_val",      c_val_);    
}


///////////////////////////////////////////////////////////////////////////////
/** Forward pass of the Actor-Critic network
 *  
 *  @param x        \input input tensor of Actor-Critic network   (torch::Tensor)
 */
tuple<torch::Tensor, torch::Tensor> actorCritic::forward(torch::Tensor x) {
        
    // Actor.
    mu_    = torch::relu(a_lin1_->forward(x));
    mu_    = torch::relu(a_lin2_->forward(mu_));
    mu_    = torch::tanh(a_lin3_->forward(mu_));

    // Critic.
    torch::Tensor c_out_;
    c_out_ = torch::relu(c_lin1_->forward(x));
    c_out_ = torch::relu(c_lin2_->forward(c_out_));
    c_out_ = torch::tanh(c_lin3_->forward(c_out_));
    c_out_ = c_val_->forward(c_out_);

    // when we do actorCriticObj.train(), torch::nn::Module method is_training() returns true
    if (this->is_training()) {
        torch::NoGradGuard no_grad;
        torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
        return make_tuple(action, c_out_);  
    } else {
        return make_tuple(mu_, c_out_);  
    }

}

///////////////////////////////////////////////////////////////////////////////
/*  Initialize network parameters
 *
 *  Initializes the model's parameters (weights and biases of the network layers)
 *  using a normal distribution of mean 'mu' and stand. dev. 'std'
 * 
 *  @param mu  (double): normal distribution mean
 *  @param std (double): normal distribution standard distribution
 */
void actorCritic::normal(double mu, double std) {
    {
        // turn off temporaly disables gradient calculations within the scope of {}
        // this avoids unnecessary memory usage when generating random numbers
        torch::NoGradGuard no_grad;

        // The for loop iterates through the parameters of the actorCritic model, taking
        // the reference (&p) to the elements of the parameters collection, of type
        // detected by automatic type inference   
        for (auto &p: this->parameters()) {
            // set random value, generated using a normal distribution with mu, std
            p.normal_(mu, std);
        }         
    }
}


///////////////////////////////////////////////////////////////////////////////
/** Differential entropy of normal distribution
 * 
 *  Calculates the differential entropy of a normal distribution. 
 *  For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
 */
torch::Tensor actorCritic::entropy() {
    return 0.5 + 0.5*log(2*M_PI) + log_std_;    // M_PI is the pi value provided by math.h class
}


///////////////////////////////////////////////////////////////////////////////
/** Logarithmic probability of taken action, given the current distribution.
 * 
 * @param action    \input action of which the logarithmic probability is calculated 
 */
torch::Tensor actorCritic::log_prob(torch::Tensor action) {
    torch::Tensor var = (log_std_ + log_std_).exp();
    return -((action - mu_)*(action - mu_))/(2*var) - log_std_ - log(sqrt(2*M_PI));
} 

// TODO: if compilation doesn't work, maybe there is an issue with the way
// libtorch manages the class/modules. rename actorCritic as actorCriticImpl, and 
// uncomment this line:
// TORCH_MODULE(actorCritic)