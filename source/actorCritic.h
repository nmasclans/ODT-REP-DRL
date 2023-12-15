/**
 * @file actorCritic.h
 * @brief Header file for class \ref actorCritic
 */

#pragma once

#include <string>
#include <torch/torch.h>
#include <vector>

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** @brief Class implementing `actorCritic` object
 *
 *  @author Nuria Masclans
 */

class actorCritic : public torch::nn::Module {


    //////////////////// DATA MEMBERS //////////////////////
    
    public: 

        // Actor
        torch::nn::Linear a_lin1_, a_lin2_, a_lin3_;
        torch::Tensor     mu_, log_std_;
        
        // Critic
        torch::nn::Linear c_lin1_, c_lin2_, c_lin3_, c_val_;

    //////////////////// MEMBER FUNCTIONS /////////////////

        tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
        void          normal(double mu, double std);
        torch::Tensor entropy();
        torch::Tensor log_prob(torch::Tensor action);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        actorCritic(int input_size, int output_size, int neurons_per_layer, double std);
        void init(){};
        virtual ~actorCritic(){};

};