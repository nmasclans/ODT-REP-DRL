/**
 * @file dv_reynolds_stress.cc
 * @brief Source file for class dv_reynolds_stress
 */

#include "dv_reynolds_stress.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! dv_reynolds_stress constructor function
 *
 * @param p_domn  \input set domain pointer with.
 * @param p_phi   \input set vector pointer with.
 * channelFlow: sets L_transported = true, L_output = true
 */

dv_reynolds_stress::dv_reynolds_stress() {

    // -> eigenvalues -> barycentric map - coordinates (2 dof)
    x1c     = vector<double>{1.0, 0.0};                               // corner x1c
    x2c     = vector<double>{0.0, 0.0};                               // corner x2c
    x3c     = vector<double>{0.5, sqrt(3.0) / 2.0};                   // corner x3c
    
    // Direct mapping:  xmap = B * eigenvalues + b, where b = x3c
    b = vector<double>(2, 0.0);
    vector<vector<double>> B(2, vector<double>(2, 0.0));
    for (int i=0; i<2; i++){
        b[i]     =   x3c[i];
        B[i][0]  =   x1c[i] + 2 * x2c[i] - 3 * x3c[i];
        B[i][1]  = - x1c[i] + 4 * x2c[i] - 3 * x3c[i];
    }

    // Inverse mapping: eigenvalues = B^{-1} * (xmap - b), where B^{-1}:
    double Bdet  = B[0][0] * B[1][1] - B[0][1] * B[1][0];
    Binv         = vector<vector<double>>(2, vector<double>(2, 0.0));
    Binv[0][0]   =   B[1][1] / Bdet;
    Binv[0][1]   = - B[0][1] / Bdet;
    Binv[1][0]   = - B[1][0] / Bdet;
    Binv[1][1]   =   B[0][0] / Bdet;

    // Baricentric coordinates of xmap triangle cartesian coordinates, source: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    // lambda1,2 = Tinv * (xmap - t); lambda3 = 1 - lambda1 - lambda2
    t    = vector<double>(2, 0.0);
    vector<vector<double>> T(2, vector<double>(2, 0.0));
    Tinv = vector<vector<double>>(2, vector<double>(2, 0.0));
    for (int i=0; i<2; i++){
        t[i]    = x3c[i];
        T[i][0] = x1c[i] - x3c[i];
        T[i][1] = x2c[i] - x3c[i];
    }
    double Tdet = T[0][0] * T[1][1] - T[0][1] * T[1][0];
    Tinv[0][0]  =   T[1][1] / Tdet;
    Tinv[0][1]  = - T[0][1] / Tdet;
    Tinv[1][0]  = - T[1][0] / Tdet;
    Tinv[1][1]  =   T[0][0] / Tdet;

}

// Direct barycentric mapping: from eigenvalues to barycentric coordinates
void dv_reynolds_stress::getDirectBarycentricMapping(const vector<vector<double>> &Dij, double &xmapping1, double &xmapping2){
    xmapping1 = x1c[0] * (    Dij[0][0] - Dij[1][1]) \
              + x2c[0] * (2.0*Dij[1][1] - 2.0*Dij[2][2]) \
              + x3c[0] * (3.0*Dij[2][2] + 1.0);
    xmapping2 = x1c[1] * (    Dij[0][0] - Dij[1][1]) \
              + x2c[1] * (2.0*Dij[1][1] - 2.0*Dij[2][2]) \
              + x3c[1] * (3.0*Dij[2][2] + 1.0);
}


// Inverse barycentric mapping: from barycentric coordinates to eigenvalues
void dv_reynolds_stress::getInverseBarycentricMapping(const double &xmapping1, const double &xmapping2, vector<vector<double>> &Dij){
    vector<double> xmapping = {xmapping1, xmapping2};
    for (int i = 0; i < 2; i++){
        Dij[i][i] = 0.0;
        for (int j = 0; j < 2; j++){
            Dij[i][i] += Binv[i][j] * (xmapping[j] - b[j]);
        }
    }
    Dij[2][2] = - Dij[0][0] - Dij[1][1]; // by constrain: sum(eigenvalues) = 0, Emory2013-A
}

// Convert cartesian coordinates xmapping to barycentric coordinates lambda, both from the barycentric map
/* xmapping1,2: cartesian coordinates of barycentric map
   lambda1,2,3: barycentric coordinates of barycentric map, satifying lambda1+lambda2+lambda3=1
   conversion:  lambda1,2 = Tinv * (xmapping_1,2 - t) 
 */
void dv_reynolds_stress::getBarycentricCoordFromCartesianCoord(const double &xmapping1, const double &xmapping2, vector<double> &lambda){
    vector<double> xmapping = {xmapping1, xmapping2};
    if (lambda.size() != 3)
        throw invalid_argument("Lambda vector must have a length of 3.");
    // construct lambda barycentric coordinates
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            lambda[i] += Tinv[i][j] * (xmapping[j] - t[j]);
        }
    }
    lambda[2] = 1.0 - lambda[0] - lambda[1];
}

void dv_reynolds_stress::getCartesianCoordFromBarycentricCoord(const vector<double> &lambda, double &xmapping1, double &xmapping2){
    if (lambda.size() != 3)
        throw invalid_argument("Lambda vector must have a length of 3.");
    // construct lambda barycentric coordinates
    xmapping1 = lambda[0] * x1c[0] + lambda[1] * x2c[0] + lambda[2] * x3c[0];
    xmapping2 = lambda[0] * x1c[1] + lambda[1] * x2c[1] + lambda[2] * x3c[1];
}

// truncates each coordinate to be within [0,1] range, and
// normalizes the coordinates vector to have the sum of elements = 1
void dv_reynolds_stress::truncateAndNormalizeBarycentricCoord(vector<double> &lambda){
    if (lambda.size() != 3)
        throw invalid_argument("Lambda vector must have a length of 3.");
    // truncate coordinates
    for (int i = 0; i < 3; i++)
        lambda[i] = std::min(std::max(lambda[i], 0.0), 1.0); 
    // normalize coordinates vector to sum(coordinates) = 1
    double sumCoord = lambda[0] + lambda[1] + lambda[2];
    for (int i = 0; i < 3; i++)
        lambda[i] /= sumCoord; 
}


// Check realizability of xmap coordinates (of eigenvalues) by:
/* 1st. transforming xmap cartesian coordinates to barycentric coordinates
   2nd. xmapping is inside the barycentric map triangle (defined by x1c, x2c, x3c, thus realizable) 
        iff lambda_i>=0 && sum_i(lambda_i)=1
   Attention: sum_i(lambda_i)=1 condition is already garanteed by construction in 'getBarycentricCoordFromCartesianCoord' method,
              only 0<=lambda_i<=1 condition is assessed.
*/
void dv_reynolds_stress::enforceRealizabilityXmap(double &xmap1, double &xmap2){
    vector<double> lambda(3, 0.0);
    getBarycentricCoordFromCartesianCoord(xmap1, xmap2, lambda); // update lambda
    if (!areElementsInRange(lambda)){ // xmap is located outside the realizable equilater traingle (defined by x1c, x2c, x3c) of the barycentric map 
        cout << "\nPerturbed xmap not realizable, outside barycentric map triangle." << endl;
        cout << "Non-realizable lambda = (" << lambda[0] << ", " << lambda[1] << ", " << lambda[2] << ")" << endl;
        truncateAndNormalizeBarycentricCoord(lambda);
        cout << "Realizable lambda = (" << lambda[0] << ", " << lambda[1] << ", " << lambda[2] << ")" << endl;
    }
    getCartesianCoordFromBarycentricCoord(lambda, xmap1, xmap2);
}

// Check all elements of vector<double> vec are in range [0.0, 1.0]
// returns true if all elements are contained within [0.0, 1.0], false otherwise
bool dv_reynolds_stress::areElementsInRange(const vector<double> &vec) {
    // Use std::all_of to check if all elements satisfy the condition
    return std::all_of(vec.begin(), vec.end(), [](double vec_i) {
        return 0.0 <= vec_i && vec_i <= 1.0;
    });
}


