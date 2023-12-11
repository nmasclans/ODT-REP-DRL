/**
 * @file eigenDecomposition.h
 * @brief Header file for class \ref eigenDecomposition
 */

#pragma once

#include <vector>
#include <string>

// class domain;    // commented for testing compilation

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing eigenDecomposition object
 *
 *  @author Núria Masclans
 */

class eigenDecomposition_ {

    public: 

    //////////////////// DATA MEMBERS //////////////////////

        //domain                   *domn;          // commented for testing compilation

    //////////////////// MEMBER FUNCTIONS /////////////////

        void sym_diagonalize(const vector<vector<double>> &A, vector<vector<double>> &Q, vector<vector<double>> &D);
        void reconstruct_matrix_from_decomposition(const vector<vector<double>> &D, const vector<vector<double>> &Q, vector<vector<double>> &A);
        void sortEigenValuesAndEigenVectors(vector<vector<double>> &Qij, vector<vector<double>> &Dij);

    private:

        void matrix_matrix_multiply(const vector<vector<double>> &A, const vector<vector<double>> &B, vector<vector<double>> &C);

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        eigenDecomposition_(){};
        // void init(domain *p_domn);       // commented for testing compilation
        void init();
        ~eigenDecomposition_(){};

};

    
////////////////////////////////////////////////////////////////////////////////
