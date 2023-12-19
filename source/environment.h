/**
 * @file environment.h
 * @brief Header file for class \ref environment
 */

#pragma once

#include <vector>
#include <string>
#include <torch/torch.h>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////
struct stepResult {
    vector<double>  reward;
    vector<double>  observation;
    bool            truncated;
    bool            terminated;
};

////////////////////////////////////////////////////////////////////////////////
/** @brief Class implementing `environment` object
 *
 *  @author Nuria Masclans
 */
class environment {


    //////////////////// DATA MEMBERS //////////////////////

    public: 

        domain         *domn;          ///< pointer to domain object

    private:

        vector<double>  x_obs;
        vector<double>  d_obs;
        vector<double>  x_act;
        vector<double>  d_act;

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        void            testTorch();
        vector<double>  reset();
        stepResult      step(const int &action_idx);


    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        environment();
        void init(domain *p_domn);
        virtual ~environment(){};

};