/**
 * @file reinforcementLearning.h
 * @brief Header file for class \ref reinforcementLearning
 */

#pragma once

#include <vector>
#include <string>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** @brief Class implementing `reinforcementLearning` object
 *
 *  @author Nuria Masclans
 */

class reinforcementLearning {


    //////////////////// DATA MEMBERS //////////////////////

    public: 

        domain         *domn;          ///< pointer to domain object

    private:

        vector<double>  x_probes;
        vector<double>  d_probes;
        vector<double>  x_act;
        vector<double>  d_act;

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        void testTorch();

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        reinforcementLearning();
        void init(domain *p_domn);
        virtual ~reinforcementLearning(){};

};