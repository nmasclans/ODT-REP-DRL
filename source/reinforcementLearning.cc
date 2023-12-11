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
  
    torch::Tensor tensor = torch::rand({2, 3});
    cout << tensor << endl;

}