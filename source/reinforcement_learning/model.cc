/**
 * @file model.cc
 * @brief Source file for class model
 */

#include "model.h"
#include "domain.h"
#include "environment.h"
#include "replayMemory.h"

#include <torch/torch.h>

using namespace std;


///////////////////////////////////////////////////////////////////////////////
/** Model initializer
 *
 * @param p_domn  \input set domain pointer with.
 */
void model::init(domain *p_domn) {
    
    domn    = p_domn;

    // get network hyperparameters 
    batch_size = domn->pram->dqnBatchSize;
    eps_start  = domn->pram->dqnEpsStart;
    eps_end    = domn->pram->dqnEpsEnd;
    eps_decay  = domn->pram->dqnEpsDecay;
    tau        = domn->pram->dqnTau;

    // construct policy & target networks
    policy_net = new dqn(domn->pram->dqnNObserv, domn->pram->dqnNActions, domn->pram->dqnNeuronsPerLayer);
    target_net = new dqn(domn->pram->dqnNObserv, domn->pram->dqnNActions, domn->pram->dqnNeuronsPerLayer);
    
    // clone policy_net into target_net
    /**Original python code:
     * load_state_dict()
     * state_dict()
     * target_net.load_state_dict(policy_net.state_dict())
     * Code transformed to C++ API (https://github.com/pytorch/pytorch/issues/36577): */
    std::stringstream stream;
    torch::save(*policy_net, stream);
    torch::load(*target_net, stream);

    // send networks to 'device'
    policy_net->to(device);
    target_net->to(device);

    // construct optimizer & memory objects
    optimizer  = new torch::optim::AdamW(
        policy_net->parameters(),
        torch::optim::AdamWOptions().lr(domn->pram->dqnLr).amsgrad(true));
    memory     = new replayMemory(10000);

    // steps counter
    steps_done = 0;

}


///////////////////////////////////////////////////////////////////////////////
/**Model constructor
 * 
*/
model::model() : device(torch::kCPU),
                 policy_net(nullptr),
                 target_net(nullptr),
                 optimizer(nullptr),
                 memory(nullptr) {

    // Device
    // check if GPU is available and set device to GPU
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

}


///////////////////////////////////////////////////////////////////////////////
/** Select Action
 *  Selection of the action to take accordingly to an epsilon greedy policy.
 *  Sometimes we'll use the policy model for choosing the action, and 
 *  sometimes we'll just sample one uniformly.
 *  The probability of choosing a random action will start at 'eps_start' and
 *  will decay exponentially towards 'eps_end'. 'eps_decay' controls the 
 *  rate of decay.
 * 
 * @param state         \input (torch::Tensor)
 * @param policy_net    \input (dqn)
 * 
 * @return index of the action choosen
 */
int64_t model::select_action(torch::Tensor state) {
     
    // rd: non-deterministic random number generator based on hardware entropy sources 
    random_device  rd;
    // mt19937 gen(rd()): mersenne twister pseudo-random number generator engine 'gen'
    // using the (random) seed from rd()
    mt19937        gen(rd());
    // uniform distribution of double values between 0 and 1
    uniform_real_distribution<double> dis(0, 1);
    // sample: single (random) sample of the uniform distribution
    double          sample = dis(gen);
    double          eps_threshold = eps_end + (eps_start - eps_end) * std::exp(-1.0 * steps_done / eps_decay);
    steps_done++;

    if (sample > eps_threshold) {
        // use policy model for chosing the action
        torch::NoGradGuard  no_grad;
        std::tuple result = policy_net->forward(state).max(1, true);   // TODO: keepdim=true as python, but consider using false
        // policy_net->forward(state) returns torch::Tensor
        // policy_net->forward(state).max(1,true) returns std::tuple
        /** result: maximum values & indices of policy_net->forward(state) along dimension 1
         *  1st arg (int)  1    : perform maximum along dimension 1
         *  2nd arg (bool) true : whether the operation should keep the dimension that was reduced as a singleton dimensional in the output
         *  i.e. if policy_net->forward(state) = torch::tensor({{0.1, 0.5, 0.3},
         *                                                      {0.2, 0.7, 0.8},
         *                                                      {0.9, 0.4, 0.7}}),
         *       then result = std::tuple({0.5,0.8,0.9},{1,2,0}),
         *       with 1st tuple element: std::get<0>(result)=torch::Tensor({0.5,0.8,0.9}) max. values along dim 1
         *            2nd tuple element: std::get<1>(result)=torch::Tensor({1,2,0}) indices of the max. values along dim 1
         */ 
        return std::get<1>(result).item<int64_t>();;
    } else {
        // sample action with (random) uniform probability
        std::uniform_int_distribution<int64_t> action_dist(0, domn->pram->dqnNActions - 1);
        return action_dist(gen);
    }

}


///////////////////////////////////////////////////////////////////////////////
/** Model optimization
 *  Performs a single step of the optimization, by:
 *      (1) sample a batch,
 *      (2) concatenate all the tensors into a single one,
 *      (3) (POLICY NETWORK) compute Q(s_t, a_t),
 *      (4) (TARGET NETWORK) compute V(s_{t+1}) = max_a q(s_{t+1},a) for added stability
 *      (4) combine Q,V into the loss
 *  * by definition, we set V(s) = 0 if s is a terminal state
 *  The network is updated at every step with a 'soft update' controlled by the 
 *  hyperparameter 'tau'.
 *  
 */
void model::optimize() {

    if (memory->size() < batch_size)
        return;

    // Sample transitions in a batch, and
    // Compute a mask of non-final states and concatenate the batch elements
    // (a final state would've been the one after which simulation ended)
    torch::Tensor non_final_mask, state_batch, action_batch, non_final_next_state_batch, reward_batch;
    memory->sample(batch_size, non_final_mask, state_batch, action_batch, non_final_next_state_batch, reward_batch);
    non_final_mask.to(device);
    state_batch.to(device);
    action_batch.to(device);
    non_final_next_state_batch.to(device);
    reward_batch.to(device);

    // Compute Q(s_t, a)
    /* The model computes Q(s_t), then we select the columns of actions taken. These are
     * the actions which would've been taken for each batch state according to policy_net
     */
    /* 'gather' selects elements from the input tensor 'state_batch_tensor' according to the indices provided 'actions_batch_tensor'
     * in this case, policy_net(state_batch_tensor) computes the Q-values for all actions the batch of states, and
     * 'action_batch_tensor' contains the indices of the actions taken in each correspoinding state, so
     * 'state_action_values' selects the Q-values of the actions taken 
     */
    torch::Tensor state_action_values = policy_net->forward(state_batch).gather(1, action_batch);

    /* Compute V(s_{t+1}) for all next states
     * Expected values of actions for non_final_next_states are computed based 
     * on the "older" target_net; selecting their best reward with max(1).values
     * This is merged based on the mask, such that we'll have either the expected
     * state value or 0 in case the state was final.
     */
    torch::Tensor next_state_values = torch::zeros({domn->pram->dqnBatchSize}, torch::kFloat32);
    {
        torch::NoGradGuard no_grad;     // temporally disables gradient calculation within {} 
        // 'target_values' are the maximum Q-value for each next state using the target network
        // (syntax error) torch::Tensor target_values = target_net->forward(non_final_next_state_batch).max(1).values;  
        torch::Tensor target_values = std::get<0>(target_net->forward(non_final_next_state_batch).max(1, false)); // used default python argument keepdim=false
        // update 'next_state_values' tensor using indexes provided by 'non_final_mask_tensor' 
        // with the values from 'target_values' 
        next_state_values.index_put_({non_final_mask}, target_values);
    }

    // Compute expected Q values
    torch::Tensor expected_state_action_values = (next_state_values * domn->pram->dqnGamma) + reward_batch;

    // Compute Huber loss
    // 'unsqueeze(1)' adds a singleton dimension (of size 1) at the specified position (1 in this case)
    torch::nn::SmoothL1Loss criterion;
    torch::Tensor loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)); 

    // Optimize the model
    optimizer->zero_grad();      // clears the gradients of all optimized tensors
    loss.backward();            // computes gradients of loss w.r.t. model parameters
    torch::nn::utils::clip_grad_value_(policy_net->parameters(), 100);  // clips gradient norms to prevent explosion (maxim value 100?)
    optimizer->step();           // updates the model's wieghts based on computed gradients

}


///////////////////////////////////////////////////////////////////////////////
void model::train(int num_episodes) {

    torch::Tensor   state, action, next_state, reward;
    vector<int>     episode_durations;
    vector<double>  state_;
    stepResult      step_result;
    int64_t         action_idx;

    for (int i_episode = 0; i_episode < num_episodes; ++i_episode) {
        
        // Initialize the environment and get its state
        // Environment initialization returns ODT to initial condition when RL is applied
        // -> get initial state
        state_ = domn->env->reset();
        // -> convert state to torch::Tensor type float32
        state  = torch::from_blob(state_.data(), {1, static_cast<long int>(state_.size())}, torch::kFloat32).clone().to(device);
        
        for (int64_t t=0; ; ++t) {

            // select action using epsilon-greedy policy
            action_idx      = select_action(state); 
            action          = torch::tensor(action_idx, torch::kInt64).clone().to(device);
            
            // perform action, advance environment
            step_result     = domn->env->step(action_idx);
            reward          = torch::from_blob(step_result.reward.data(), {1, static_cast<long int>(step_result.reward.size())}, torch::kFloat32).clone().to(device);
            
            // get next state
            if (step_result.terminated) {
                next_state  = torch::Tensor();
            } else {
                next_state  = torch::from_blob(step_result.observation.data(), {1, static_cast<long int>(step_result.observation.size())}, torch::kFloat32).clone().to(device);
            }
            
            // store transition in memory
            memory->push(state, action, next_state, reward);
            
            // move to the next state (update next_state)
            state = next_state.clone();
            
            // perform one optimization step (on the policy network, policy_net)
            optimize();

            /** soft update of the target network's weights (target_net)
             *  θ′ ← τ θ + (1 − τ ) θ′, with θ  the policy_net weights, 
             *                               θ' the target_net weights.
             * -> retrive the state dictionary of the policy & target networks
            /** C++ Libtorch API has not yet implemented state_dict() and load_state_dict() torch methods
             *  As a consequence, following python-based lines would not work: 
             * 
             *  // get the parameters of both networks
             *  auto policy_net_state_dict = policy_net->state_dict();
             *  auto target_net_state_dict = target_net->state_dict();
             *  // calculate updated weights of target_net using soft update strategy
             *  for (const auto &key : policy_net_state_dict.keys()) {
             *      target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau);
             *  }
             *  // update target_net
             *  target_net->load_state_dict(target_net_state_dict);
             * 
             *  Alternatively, the network parameters are accessed directly using .parameters().
             */
            // TODO: check if this update of parameters is correct, by:
            // TODO: print the target_net (should change according to soft-update equation) and policy_net (should remain constant) prior and after the target_net update.
            // get the parameters of both networks
            vector<torch::Tensor> policy_params = policy_net->parameters();
            vector<torch::Tensor> target_params = target_net->parameters();
            // calculate updated weights of target_net using soft update strategy
            for (size_t i = 0; i < policy_params.size(); ++i) {
                // update target_net params based on policy_net params
                target_params[i].data().mul_(1 - tau).add_(policy_params[i].data().mul_(tau));
            }
            // update target_net
            for (size_t i = 0; i < target_params.size(); ++i) {
                target_net->parameters()[i].data().copy_(target_params[i].data());
            }

            if (step_result.terminated || step_result.truncated) {
                episode_durations.push_back(t + 1);
                break;
            }          
        }
    }

    // log episodes duration:
    cout << endl << "Episode durations: " << endl << episode_durations << endl;

}