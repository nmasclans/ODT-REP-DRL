/**
 * @file dqn.h
 * @brief Header file for class dqn
 */

#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing dqn (Deep Q-Learning Network) object.
 *
 *  @author Núria Masclans
 */

struct dqn : public torch::nn::Module {

    //////////////////// DATA MEMBERS //////////////////////

    private:

        torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        torch::Tensor forward(torch::Tensor x);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

        //dqn(){};
        dqn(int n_observations, int n_actions, int n_neurons_per_layer);

};