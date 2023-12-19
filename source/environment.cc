/**
 * @file environment.cc
 * @brief Source file for class \ref environment
 */

#include "environment.h"
#include "domain.h"

#include <iostream>
#include <torch/torch.h>
#include <vector>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
/** environment constructor function
 */
environment::environment() {
    
    // add code

}

///////////////////////////////////////////////////////////////////////////////
/** reinforcementLearning initialization function
 *
 * @param p_domn  \input set domain pointer with.
 */

void environment::init(domain *p_domn) {

    domn    = p_domn;

    vector<double> x_obs(domn->pram->dqnNObserv,  0.0);     // observations position 
    vector<double> d_obs(domn->pram->dqnNObserv,  0.0);     // observations data
    vector<double> x_act(domn->pram->dqnNActions, 0.0);     // actions position
    vector<double> d_act(domn->pram->dqnNActions, 0.0);     // actions data

}

///////////////////////////////////////////////////////////////////////////////
/**
 * Test torch package
 */
void environment::testTorch() {
  
    // output main CUDA information
    cout << "cuda is available: " << torch::cuda::is_available() << endl;
    cout << "number of available gpus: " << torch:: cuda::device_count() << endl;

    // check if CUDA (GPU) is available
    if (torch::cuda::is_available()){
        
        // create a CUDA device
        torch::Device device(torch::kCUDA);
        // you could alternatively specify the targeted gpu, identified by an input variable gpu_id (int):
        /*
        if (gpu_id >= torch::cuda::device_count()) {
            cout << "Invalid GPU ID. Using default GPU (ID 0)." << endl;
            gpu_id = 0;
        }
        torch::Device device(torch::kCUDA, gpu_id);
        */

        // create & move tensor to the CUDA device
        torch::Tensor tensor = torch::rand({2, 3}).to(device);

        // print tensor (will be on GPU)
        cout << tensor << endl;

    } else {
        cout << "CUDA is not available. Running on CPU." << endl;
    }

}



///////////////////////////////////////////////////////////////////////////////
/**
 * 
*/
vector<double> environment::reset() {

    vector<double state;

    // TODO: add content

    return state;
    
} 


///////////////////////////////////////////////////////////////////////////////
/**
 * 
*/
stepResult environment::step(const int &action_idx){

    vector<double> reward, observation;
    bool           truncated, terminated;
    
    // TODO: add content

    return {reward, observation, truncated, terminated}

}



