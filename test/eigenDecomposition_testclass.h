/**
 * @file eigenDecomposition_testclass.h
 * @brief Header file for class \ref eigenDecomposition
 */

#pragma once

#include <vector>

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing eigenDecomposition object
 *
 *  @author Núria Masclans
 */

class eigenDecomposition_testclass {

    public: 

    //////////////////// DATA MEMBERS //////////////////////

    //////////////////// MEMBER FUNCTIONS /////////////////

        void sym_diagonalize(const double (&A)[3][3], double (&Q)[3][3], double (&D)[3][3]);
        void reconstruct_matrix_from_decomposition(const double (&D)[3][3], const double (&Q)[3][3], double (&A)[3][3]);

    private:

        void matrix_matrix_multiply(const double (&A)[3][3], const double (&B)[3][3], double (&C)[3][3]);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        eigenDecomposition_testclass(){};
        void init();
        ~eigenDecomposition_testclass(){};

};

    
////////////////////////////////////////////////////////////////////////////////
