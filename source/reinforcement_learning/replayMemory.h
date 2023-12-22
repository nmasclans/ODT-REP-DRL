#pragma once

#include <torch/torch.h>
#include <tuple>
#include <vector>

class domain;

using namespace std;

/**Transition:
 * Object representing a single transition in the environment. It maps
 * (state, action) pairs to their (next_state, reward) result.
 */
struct Transition {
    torch::Tensor state;
    torch::Tensor action;
    torch::Tensor next_state;
    torch::Tensor reward;

    // Constructor to initialize Transition directly from tensors
    /* in c++ LibTorch API, .clone() already performs python .detach(),
     * as the cloned tensor doesn't retain any gradient information by default,
     * similar to PyTorch. Thus, it is not necessary to explicitely use .detach().
     */
    Transition(torch::Tensor state_, torch::Tensor action_, torch::Tensor next_state_, torch::Tensor reward_) :
        state(move(state_)),
        action(move(action_)), 
        next_state(move(next_state_)),
        reward(move(reward_)) {}

};

/**Replay Memory:
 * The experience replay memory will be used for training the DQN.
 * It stores the transitions that the agent observes, allowing us to
 * reuse this data later. By sampling from the replay memory randomly,
 * the transitions that build up a batch are decorrelated. It has
 * been shown that this greatly stabilizes and improves the DQN
 * training procedure. 
 */
class replayMemory {

    //////////////////// DATA MEMBERS //////////////////////

    private:
        
        size_t capacity_;
        deque<Transition> memory_;
        /** 'deque' (instead of 'vector'):
         * The choice between std::deque and std::vector depends on the requirements of the
         * application. Both containers have different performance characteristics, especially
         * regarding insertion and deletion of elements. std::deque is generally efficient for
         * fast insertion and deletion at both ends (push_front, push_back, pop_front, pop_back),
         * while std::vector has more efficient element access (random access).
         * For a replay memory used in reinforcement learning, where you usually append new 
         * transitions and might remove the oldest ones, std::deque is a reasonable choice.
         */

    //////////////////// MEMBER FUNCTIONS /////////////////
    
    public:
    
        void    push(torch::Tensor state, torch::Tensor action, torch::Tensor next_state, torch::Tensor reward);

        size_t  size() const;
        // 'const': the method does not modify the internal state of the object;
        // calling size() won't change any member variables of the 'replayMemory' instance it's called on

        void    sample(const size_t  &batch_size,
                       torch::Tensor &non_final_mask, 
                       torch::Tensor &state_batch,
                       torch::Tensor &action_batch,
                       torch::Tensor &non_final_next_state_batch,
                       torch::Tensor &reward_batch);

    private:

        vector<Transition>  sample_vector_of_transitions(const size_t &batch_size);



    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        explicit replayMemory(size_t capacity);

};