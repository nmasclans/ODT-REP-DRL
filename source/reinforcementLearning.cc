/**
 * @file reinforcementLearning.cc
 * @brief Source file for class \ref reinforcementLearning
 */

#include "reinforcementLearning.h"
#include "domain.h"

#include <torch/torch.h>
#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
/** micromixer constructor function
 */

reinforcementLearning::reinforcementLearning() {
    
    // add code

}

///////////////////////////////////////////////////////////////////////////////
/** reinforcementLearning initialization function
 *
 * @param p_domn  \input set domain pointer with.
 */

void reinforcementLearning::init(domain *p_domn) {

    domn    = p_domn;

}

///////////////////////////////////////////////////////////////////////////////
/**
 * Test torch package
*/
void reinforcementLearning::testTorch() {
  
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