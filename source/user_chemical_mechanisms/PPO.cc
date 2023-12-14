/**
 * @file PPO.cc
 * @brief Source file for class \ref PPO
 */

#include "PPO.h"
#include "actorCritic.h"

#include <torch/torch.h>
#include <random>

using namespace std;
using VT = std::vector<torch::Tensor>;      // Vector of tensors.
using OPT = torch::optim::Optimizer;    // Optimizer

// Random engine for shuffling memory.
std::random_device rd;
std::mt19937 re(rd());

///////////////////////////////////////////////////////////////////////////////
/** Returns method by Generalized Advantage Estimate
 * 
 * Reference: Generalized Advantage Estimate, https://arxiv.org/abs/1506.02438
 */
VT PPO::returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda)
{
    
    // Compute the returns.
    torch::Tensor gae = torch::zeros({1}, torch::kFloat64);
    VT returns(rewards.size(), torch::zeros({1}, torch::kFloat64));

    // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
    for (uint i=rewards.size(); --i >= 0;){
        // Advantage.
        auto delta = rewards[i] + gamma*vals[i+1]*(1-dones[i]) - vals[i];
        gae = delta + gamma*lambda*(1-dones[i])*gae;

        returns[i] = gae + vals[i];
    }

    return returns;
}

void PPO::update(actorCritic& ac,
                 torch::Tensor& states,
                 torch::Tensor& actions,
                 torch::Tensor& log_probs,
                 torch::Tensor& returns,
                 torch::Tensor& advantages, 
                 OPT& opt, 
                 uint steps, uint epochs, uint mini_batch_size, double beta, double clip_param)
{
    
    for (uint e=0;e<epochs;e++) {
        // Generate random indices.
        torch::Tensor cpy_sta = torch::zeros({mini_batch_size, states.size(1)}, states.options());
        torch::Tensor cpy_act = torch::zeros({mini_batch_size, actions.size(1)}, actions.options());
        torch::Tensor cpy_log = torch::zeros({mini_batch_size, log_probs.size(1)}, log_probs.options());
        torch::Tensor cpy_ret = torch::zeros({mini_batch_size, returns.size(1)}, returns.options());
        torch::Tensor cpy_adv = torch::zeros({mini_batch_size, advantages.size(1)}, advantages.options());

        for (uint b=0;b<mini_batch_size;b++) {
            uint idx   = uniform_int_distribution<uint>(0, steps-1)(re);
            cpy_sta[b] = states[idx];
            cpy_act[b] = actions[idx];
            cpy_log[b] = log_probs[idx];
            cpy_ret[b] = returns[idx];
            cpy_adv[b] = advantages[idx];
        }

        auto av = ac->forward(cpy_sta); // action value pairs
        auto action = get<0>(av);
        auto entropy = ac->entropy().mean();
        auto new_log_prob = ac->log_prob(cpy_act);

        auto old_log_prob = cpy_log;
        auto ratio = (new_log_prob - old_log_prob).exp();
        auto surr1 = ratio*cpy_adv;
        auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param)*cpy_adv;

        auto val = get<1>(av);
        auto actor_loss = -torch::min(surr1, surr2).mean();
        auto critic_loss = (cpy_ret-val).pow(2).mean();

        auto loss = 0.5*critic_loss+actor_loss-beta*entropy;

        opt.zero_grad();
        loss.backward();
        opt.step();
    }

}
