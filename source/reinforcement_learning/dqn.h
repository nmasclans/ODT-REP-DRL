/**
 * @file dqn.h
 * @brief Header file for class dqn
 */

#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing dqn (Deep Q-Learning Network) object.
 *
 *  @author Núria Masclans
 */

class dqn : public torch::nn::Module {


    //////////////////// DATA MEMBERS //////////////////////

    public: 

        domain              *domn;              //< pointer to domain object
    
    private:

        torch::nn::Linear*   layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr };

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        torch::Tensor forward(torch::Tensor x);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

        dqn(int n_observations, int n_actions, int n_neurons_per_layer);
        void init(domain *line);
        virtual ~dqn();

};
