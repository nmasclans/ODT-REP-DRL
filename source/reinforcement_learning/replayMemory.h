#pragma once

#include <torch/torch.h>
#include <tuple>
#include <vector>

class domain;

using namespace std;

struct Transition {
    torch::Tensor state;
    torch::Tensor action;
    torch::Tensor next_state;
    torch::Tensor reward;
};

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
    
        void push(torch::Tensor state, torch::Tensor action, torch::Tensor next_state, torch::Tensor reward);

        vector<Transition> sample(size_t batch_size);

        size_t size() const;
        // 'const': the method does not modify the internal state of the object;
        // calling size() won't change any member variables of the 'replayMemory' instance it's called on


    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

        explicit replayMemory(size_t capacity);

};