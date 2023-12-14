#pragma once

#include <torch/torch.h>
#include "actorCritic.h"

using namespace std;
using VT = std::vector<torch::Tensor>;  // Vector of tensors
using OPT = torch::optim::Optimizer;    // Optimizer

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO {

    //////////////////// DATA MEMBERS //////////////////////

    
    //////////////////// MEMBER FUNCTIONS /////////////////
    
    public: 

        VT returns(VT &rewards, VT &dones, VT &vals, double gamma, double lambda); 
        void update(actorCritic& ac,
                    torch::Tensor& states,
                    torch::Tensor& actions,
                    torch::Tensor& log_probs,
                    torch::Tensor& returns,
                    torch::Tensor& advantages, 
                    OPT& opt, 
                    uint steps, uint epochs, uint mini_batch_size, double beta, double clip_param=.2);


    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////
    
    public:

        PPO();
        void init(){};
        virtual ~PPO(){};

};