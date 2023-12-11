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

    public: 

    //////////////////// DATA MEMBERS //////////////////////

        domain         *domn;          ///< pointer to domain object

    //////////////////// MEMBER FUNCTIONS /////////////////

        void testTorch();

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        reinforcementLearning();
        void init(domain *p_domn);
        virtual ~reinforcementLearning(){};

};