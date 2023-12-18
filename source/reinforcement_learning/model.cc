/**
 * @file model.cc
 * @brief Source file for class model
 */

#include "model.h"
#include "domain.h"
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
}


///////////////////////////////////////////////////////////////////////////////
/** Model constructor
 * ...
*/
model::model() {

    // get network hyperparameters 
    n_actions       = domn->pram->dqnNActions;
    n_observations  = domn->pram->dqnNObserv;
    batch_size      = domn->pram->dqnBatchSize;

    // define the device
    // initialize with CPU as default
    torch::Device device(torch::kCPU); 
    // check if GPU is available and set device to GPU
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    // define policy & target networks
    policy_net = DQN(n_observations, n_actions).to(device);
    target_net = DQN(n_observations, n_actions).to(device);
    target_net.load_state_dict(policy_net.state_dict());

    optimizer = torch::optim::AdamW(policy_net.parameters(), lr=domn->pram->dqnLr, amsgrad=True);
    memory    = replayMemory(10000);
    env       = domn->env;

    steps_done = 0;

}

///////////////////////////////////////////////////////////////////////////////
/** Select Action
 * ...
 * 
 * @param state         \input (torch::Tensor)
 * @param policy_net    \input (DQN)
 */
int model::select_action(torch::Tensor state) {
     
    // rd: non-deterministic random number generator based on hardware entropy sources 
    random_device  rd;
    // mt19937 gen(rd()): mersenne twister pseudo-random number generator engine 'gen'
    // using the (random) seed from rd()
    mt19937        gen(rd());
    // uniform distribution of double values between 0 and 1
    uniform_real_distribution<double> dis(0, 1);
    // sample: single (random) sample of the uniform distribution
    double          sample = dis(gen);
    double          eps_threshold = domn->pram->dqnEpsEnd + (domn->pram->dqnEpsStart - domn->pram->dqnEpsEnd) * std::exp(-1.0 * steps_done / EPS_DECAY);
    steps_done++;

    if (sample > eps_threshold) {
        torch::NoGradGuard no_grad;
        auto result = policy_net(state).max(1, true);
        return result[1].item<int>();
    } else {
        std::uniform_int_distribution<int> action_dist(0, n_actions - 1);
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

    if (memory.size() < batch_size)
        return;

    // Sample transitions
    vector<Transition> transitions = sample(memory, batch_size);
    /* Transpose batch of transitions:
     * This converts batch-array of Transitions to Transition of batch-arrays
     * see https://stackoverflow.com/a/19343/3343043 for detailed explanation)
     */
    Transition batch = transpose(transitions);

    // Compute a mask of non-final states and concatenate the batch elements
    // (a final state would've been the one after which simulation ended)
    vector<bool>          non_final_mask;
    vector<torch::Tensor> non_final_next_states;
    vector<torch::Tensor> state_batch;
    vector<torch::Tensor> action_batch;
    vector<torch::Tensor> reward_batch;

    // fill the vectors of tensors along the batches
    for (size_t i = 0; i < batch.next_state.size(); ++i) {
        if (!batch.next_state[i].defined()) {
            non_final_mask.push_back(false);
        } else {
            non_final_mask.push_back(true);
            non_final_next_states.push_back(batch.next_state[i]);
        }
        state_batch.push_back(batch.state[i]);
        action_batch.push_back(batch.action[i]);
        reward_batch.push_back(batch.reward[i]);
    }
    
    // transform to torch:tensors
    torch::Tensor non_final_mask_tensor = torch::tensor(non_final_mask, torch::kBool);  // creates a boolean tensor from non_final_mask vector
    // 'cat' concatenates a vector of tensors along the (default) 0 dimension
    torch::Tensor non_final_next_states_tensor = torch::cat(non_final_next_states);     
    torch::Tensor state_batch_tensor    = torch::cat(state_batch);                      
    torch::Tensor action_batch_tensor   = torch::cat(action_batch);
    torch::Tensor reward_batch_tensor   = torch::cat(reward_batch);

    // Compute Q(s_t, a)
    /* The model computes Q(s_t), then we select the columns of actions taken. These are
     * the actions which would've been taken for each batch state according to policy_net
     */
    /* 'gather' selects elements from the input tensor 'state_batch_tensor' according to the indices provided 'actions_batch_tensor'
     * in this case, policy_net(state_batch_tensor) computes the Q-values for all actions the batch of states, and
     * 'action_batch_tensor' contains the indices of the actions taken in each correspoinding state, so
     * 'state_action_values' selects the Q-values of the actions taken 
     */
    torch::Tensor state_action_values = policy_net(state_batch_tensor).gather(1, action_batch_tensor);

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
        torch::Tensor target_values = target_net(non_final_next_states_tensor).max(1).values;  
        // update 'next_state_values' tensor using indexes provided by 'non_final_mask_tensor' 
        // with the values from 'target_values' 
        next_state_values.index_put_({non_final_mask_tensor}, target_values);
    }

    // Compute expected Q values
    torch::Tensor expected_state_action_values = (next_state_values * domn->pram->dqnGamma) + reward_batch_tensor;

    // Compute Huber loss
    // 'unsqueeze(1)' adds a singleton dimension (of size 1) at the specified position (1 in this case)
    torch::nn::SmoothL1Loss criterion;
    torch::Tensor loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)); 

    // Optimize the model
    optimizer.zero_grad();      // clears the gradients of all optimized tensors
    loss.backward();            // computes gradients of loss w.r.t. model parameters
    torch::nn::utils::clip_grad_value_(policy_net->parameters(), 100);  // clips gradient norms to prevent explosion (maxim value 100?)
    optimizer.step();           // updates the model's wieghts based on computed gradients

}

///////////////////////////////////////////////////////////////////////////////
void model::plot_durations(const std::vector<int> &episode_durations, bool show_result) {

    // Code for plotting the episode durations (use your preferred plotting library)
    // Implementation based on matplotlib or other plotting libraries
    // TODO: implement code
    
}


///////////////////////////////////////////////////////////////////////////////
void model::train(int num_episodes) {

    for (int i_episode = 0; i_episode < num_episodes; ++i_episode) {
        
        // Initialize the environment and get its state
        torch::Tensor state = env.reset(); // Assuming 'env' is an instance of the environment
        
        for (int64_t t=0; ; ++t) {

            // select action using epsilon-greedy policy
            int action = select_action(state, policy_net);

            CONTINUE HERE!
        }
    }

}
// TODO: transform train_loop python code bellow to a method of model class:

// Below, you can find the main training loop. At the beginning we reset the environment and obtain the initial state Tensor. Then, we sample an action, execute it, observe the next state and the reward (always 1), and optimize our model once. When the episode ends (our model fails), we restart the loop.

// Below, num_episodes is set to 600 if a GPU is available, otherwise 50 episodes are scheduled so training does not take too long. However, 50 episodes is insufficient for to observe good performance on CartPole. You should see the model constantly achieve 500 steps within 600 training episodes. Training RL agents can be a noisy process, so restarting training can produce better results if convergence is not observed.

// if torch.cuda.is_available():
//     num_episodes = 600
// else:
//     num_episodes = 50

// for i_episode in range(num_episodes):
//     # Initialize the environment and get it's state
//     state, info = env.reset()
//     state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
//     for t in count():
//         action = select_action(state)
//         observation, reward, terminated, truncated, _ = env.step(action.item())
//         reward = torch.tensor([reward], device=device)
//         done = terminated or truncated

//         if terminated:
//             next_state = None
//         else:
//             next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

//         # Store the transition in memory
//         memory.push(state, action, next_state, reward)

//         # Move to the next state
//         state = next_state

//         # Perform one step of the optimization (on the policy network)
//         optimize_model()

//         # Soft update of the target network's weights
//         # θ′ ← τ θ + (1 −τ )θ′
//         target_net_state_dict = target_net.state_dict()
//         policy_net_state_dict = policy_net.state_dict()
//         for key in policy_net_state_dict:
//             target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
//         target_net.load_state_dict(target_net_state_dict)

//         if done:
//             episode_durations.append(t + 1)
//             plot_durations()
//             break

// print('Complete')
// plot_durations(show_result=True)
// plt.ioff()
// plt.show()
