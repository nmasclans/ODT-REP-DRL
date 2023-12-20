#pragma once

#include <torch/torch.h>
#include <cmath>
#include <random>
#include <vector>

#include "dqn.h"
#include "replayMemory.h"

class domain;

using namespace std;


////////////////////////////////////////////////////////////////////////////////

/** Class implementing reinforcement learning Model object.
 *
 *  @author Núria Masclans
 */

class model {


    //////////////////// DATA MEMBERS //////////////////////

    public: 

        domain             *domn;              //< pointer to domain object
        
    private:

        torch::Device       device;
        dqn                 policy_net;
        dqn                 target_net;
        torch::optim::AdamW optimizer;
        replayMemory        memory;
        int                 n_actions;
        int                 n_observations;
        int                 n_neurons_per_layer;
        int                 batch_size;
        int                 steps_done;
        double              eps_start;
        double              eps_end;
        double              eps_decay;
        double              tau;

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        void    train(int num_episodes);

    private:

        int     select_action(torch::Tensor state); //, dqn &policy_net);
        void    optimize(); // (vector<Transition> &memory, dqn &policy_net, dqn &target_net);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

        model();
        void init(domain *line);
        virtual ~model(){};

};
