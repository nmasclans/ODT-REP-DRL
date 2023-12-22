#include "replayMemory.h"
#include <random>

using namespace std;


///////////////////////////////////////////////////////////////////////////////
replayMemory::replayMemory(size_t capacity) : capacity_(capacity) {
    // constructor code here
};


///////////////////////////////////////////////////////////////////////////////
void replayMemory::push(torch::Tensor state, torch::Tensor action, torch::Tensor next_state, torch::Tensor reward) {

    // create new transition
    Transition new_transition(state, action, next_state, reward);
    // add new transition into memory
    memory_.push_back(new_transition);

    // check memory size - compare to chosen capacity
    // -> if the memory exceeds the capacity, remove the oldest element
    if (memory_.size() >= capacity_) {
        memory_.pop_front();
    }
    
}


///////////////////////////////////////////////////////////////////////////////
vector<Transition> replayMemory::sample_vector_of_transitions(const size_t &batch_size) {
    vector<Transition> batch;
    std::sample(memory_.begin(), memory_.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});
    /* - 'std::sample' (C++17): random sampling of elements from a sequence, in this case 
     *   it samples elements from the range defined by memory_.begin() to memory_.end() 
     *   samples elements from all elements in memory_.
     * - 'memory_.begin()', 'memory_.end()': iterators referring to the beginning and end 
     *   of the 'memory_' container. 
     * - 'std::back_inserter(batch)' is an iterator that allows inserting elements at the
     *   end of the 'batch' container
     * - 'batch_size': number of elements to be (randomly) sampled from 'memory_'
     * - 'std::mt19937{std::random_device{}()}': creates a random number generation engine 
     *   ('std::mt19937') seeded with a random value from 'std::random_device'. This generator
     *   is used for random sampling 
     */
    return batch;

}

///////////////////////////////////////////////////////////////////////////////
/** Transpose batch of transitions:
 * This converts batch-array of Transitions ('sample_' output) to Transition of batch-arrays ('sample' output)
 * see https://stackoverflow.com/a/19343/3343043 for detailed explanation)
 */
// Transition replayMemory::sample_transition_of_vectors(const size_t  &batch_size) {
    
//     vector<Transition> vector_of_transitions = sample_vector_of_transitions(batch_size);

//     size_t state_size          = vector_of_transitions[0].state.size(0);
//     size_t action_size         = vector_of_transitions[0].action.size(0);
//     size_t next_state_size     = vector_of_transitions[0].next_state.size(0);
//     size_t reward_size         = vector_of_transitions[0].reward.size(0);
//     // TODO: is size the same for all 4 tensors?

//     torch::Tensor states       = torch::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(state_size)},      torch::kFloat32);
//     torch::Tensor actions      = torch::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(action_size)},     torch::kInt64);
//     torch::Tensor next_states  = torch::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(next_state_size)}, torch::kFloat32);
//     torch::Tensor rewards      = torch::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(reward_size)},     torch::kFloat32);

//     for (size_t i = 0; i < batch_size; i++) {
//         states[i]      = vector_of_transitions[i].state;
//         actions[i]     = vector_of_transitions[i].action;
//         next_states[i] = vector_of_transitions[i].next_state;
//         rewards[i]     = vector_of_transitions[i].reward;
//     }

//     Transition transition_of_vectors(states, actions, next_states, rewards);

//     return transition_of_vectors;

// }


///////////////////////////////////////////////////////////////////////////////
/**Sample random transitions, and organize in seperate vectors:
 * 
 * non_final_mask : vector storing boolean values indicating wether the next
 *      state is defined ('true') or not ('false') 
 * non_final_next_states, state_batch, action_batch, reward_batch :
 *      separate vectors storing tensors for non-final next-states, actions, 
 *      and rewards, respectively 
 */
void replayMemory::sample(const size_t  &batch_size, 
                          torch::Tensor &non_final_mask, 
                          torch::Tensor &state_batch,
                          torch::Tensor &action_batch,
                          torch::Tensor &non_final_next_state_batch,
                          torch::Tensor &reward_batch) {

    vector<Transition>    batch = sample_vector_of_transitions(batch_size);

    vector<int>           v_non_final_mask;
    vector<torch::Tensor> v_state_batch;
    vector<torch::Tensor> v_action_batch;
    vector<torch::Tensor> v_non_final_next_state_batch;
    vector<torch::Tensor> v_reward_batch;

    // fill the vectors of tensors along the batches, with each batch item a transition
    for (const Transition& transition : batch) {

        if (!transition.next_state.defined()) {
            v_non_final_mask.push_back(1);
        } else {
            v_non_final_mask.push_back(0);
            v_non_final_next_state_batch.push_back(transition.next_state);
        }

        v_state_batch.push_back(transition.state);
        v_action_batch.push_back(transition.action);
        v_reward_batch.push_back(transition.reward);
    }

    // tensor concanation
    non_final_mask              =  torch::tensor(v_non_final_mask, torch::kBool).to(torch::kBool);
    state_batch                 =  torch::cat(v_state_batch).to(torch::kFloat32);
    action_batch                =  torch::cat(v_action_batch).to(torch::kInt64);
    non_final_next_state_batch  =  torch::cat(v_non_final_next_state_batch).to(torch::kFloat32);
    reward_batch                =  torch::cat(v_reward_batch).to(torch::kFloat32);

}


///////////////////////////////////////////////////////////////////////////////
size_t replayMemory::size() const {
    return memory_.size();
}
