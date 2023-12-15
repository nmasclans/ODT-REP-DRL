#pragma once

#include <torch/torch.h>
#include <cmath>
#include <random>
#include <vector>

#include "dqn.h"

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

        domain              *domn;              //< pointer to domain object
    
    private:

        dqn                 policy_net;
        dqn                 target_net;
        torch::optim::AdamW optimizer;
        ReplayMemory        memory;
        int                 n_actions;
        int                 n_observations;
        int                 batch_size;
        int                 steps_done;


    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

    private:

        void    plot_durations(const vector<int> &episode_durations, bool show_result = false);
        int     select_action(torch::Tensor state, dqn &policy_net);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

        model();
        void init(domain *line);
        virtual ~model(){};

};
