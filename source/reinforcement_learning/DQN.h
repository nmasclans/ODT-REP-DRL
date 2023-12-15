/**
 * @file DQN.h
 * @brief Header file for class DQN
 */

#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing DQN (Deep Q-Learning Network) object.
 *
 *  @author Núria Masclans
 */

class DQN : public torch::nn::Module {


    //////////////////// DATA MEMBERS //////////////////////

    public: 

        domain              *domn;              //< pointer to domain object
    
    private:

        torch::nn::Linear*   layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr };

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        torch::Tensor forward(torch::Tensor x);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

        DQN(int64_t n_observations, int64_t n_actions, int64_t n_neurons_per_layer);
        void init(domain *line);
        virtual ~DQN();

};
