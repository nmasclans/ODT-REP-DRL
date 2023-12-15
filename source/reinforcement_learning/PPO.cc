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

void PPO::update(actorCritic   *ac,
                 torch::Tensor &states,
                 torch::Tensor &actions,
                 torch::Tensor &log_probs,
                 torch::Tensor &returns,
                 torch::Tensor &advantages, 
                 OPT           &opt, 
                 uint steps, uint epochs, uint mini_batch_size, double beta, double clip_param)
{
    
    for (uint e=0;e<epochs;e++) {
        // Generate random indices.
        torch::Tensor cpy_sta = torch::zeros({mini_batch_size, states.size(1)},     states.options());
        torch::Tensor cpy_act = torch::zeros({mini_batch_size, actions.size(1)},    actions.options());
        torch::Tensor cpy_log = torch::zeros({mini_batch_size, log_probs.size(1)},  log_probs.options());
        torch::Tensor cpy_ret = torch::zeros({mini_batch_size, returns.size(1)},    returns.options());
        torch::Tensor cpy_adv = torch::zeros({mini_batch_size, advantages.size(1)}, advantages.options());

        for (uint b=0;b<mini_batch_size;b++) {
            uint idx   = uniform_int_distribution<uint>(0, steps-1)(re);
            cpy_sta[b] = states[idx];
            cpy_act[b] = actions[idx];
            cpy_log[b] = log_probs[idx];
            cpy_ret[b] = returns[idx];
            cpy_adv[b] = advantages[idx];
        }

        tuple<torch::Tensor, torch::Tensor> av = ac->forward(cpy_sta); // action value pairs
        torch::Tensor action  = get<0>(av);
        torch::Tensor entropy = ac->entropy().mean();
        torch::Tensor new_log_prob = ac->log_prob(cpy_act);

        torch::Tensor old_log_prob = cpy_log;
        torch::Tensor ratio = (new_log_prob - old_log_prob).exp();
        torch::Tensor surr1 = ratio * cpy_adv;
        torch::Tensor surr2 = torch::clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * cpy_adv; 
        // torch::clamp restricts the values of ratio between the value range (1.0-clip_param, 1+clip_param)

        torch::Tensor val = get<1>(av);
        torch::Tensor actor_loss = - torch::min(surr1, surr2).mean();
        // torch::min performs element-wise comparison between tensors surr1, surr2, and returns the element-wise min
        torch::Tensor critic_loss = (cpy_ret-val).pow(2).mean();
        // pow() element-wise power 2, and mean() returns single element torch::Tensor of the mean value
        torch::Tensor loss = 0.5 * critic_loss + actor_loss - beta * entropy;

        opt.zero_grad();
        loss.backward();
        opt.step();
    }

}
