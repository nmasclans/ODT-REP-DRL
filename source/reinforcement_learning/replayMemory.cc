#include "replayMemory.h"
#include <random>

using namespace std;


///////////////////////////////////////////////////////////////////////////////
replayMemory::replayMemory(size_t capacity) : capacity_(capacity) {
    // constructor code here
};


///////////////////////////////////////////////////////////////////////////////
void replayMemory::push(torch::Tensor state, torch::Tensor action, torch::Tensor next_state, torch::Tensor reward) {

    // add new transition into memory
    memory_.push_back({state, action, next_state, reward});

    // check memory size - compare to chosen capacity
    // -> if the memory exceeds the capacity, remove the oldest element
    if (memory_.size() >= capacity_) {
        memory_.pop_front();
    }
    
}


///////////////////////////////////////////////////////////////////////////////
vector<Transition> replayMemory::sample(size_t batch_size) {
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
size_t replayMemory::size() const {
    return memory_.size();
}
