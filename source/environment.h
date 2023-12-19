/**
 * @file environment.h
 * @brief Header file for class \ref environment
 */

#pragma once

#include <vector>
#include <string>

class domain;

using namespace std;

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

        void testTorch();

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        environment();
        void init(domain *p_domn);
        virtual ~environment(){};

};