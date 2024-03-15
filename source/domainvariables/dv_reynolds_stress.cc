/**
 * @file dv_reynolds_stress.cc
 * @brief Source file for class dv_reynolds_stress
 */

#include "dv_reynolds_stress.h"
#include "domain.h"
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

dv_reynolds_stress::dv_reynolds_stress(domain    *line,
                                       const      string s,
                                       const bool Lt,
                                       const bool Lo) : dv(line, s, Lt, Lo) {

    L_output_stat   = true;

    // Reynolds stress terms
    Rxx     = vector<double>(nunif, 0.0);
    Ryy     = vector<double>(nunif, 0.0);
    Rzz     = vector<double>(nunif, 0.0);
    Rxy     = vector<double>(nunif, 0.0);
    Rxz     = vector<double>(nunif, 0.0);
    Ryz     = vector<double>(nunif, 0.0);

    // anisotropy tensor
    // -> trace = 2*TKE
    Rkk     = vector<double>(nunif, 0.0);
    // -> eigenvectors -> rotation angles (3 dof)
    thetaZ  = vector<double>(nunif, 0.0); 
    thetaY  = vector<double>(nunif, 0.0); 
    thetaX  = vector<double>(nunif, 0.0);
    // -> eigenvalues -> barycentric map - coordinates (2 dof)
    x1c     = vector<double>{1.0, 0.0};                               // corner x1c
    x2c     = vector<double>{0.0, 0.0};                               // corner x2c
    x3c     = vector<double>{0.5, sqrt(3.0) / 2.0};                   // corner x3c
    xmap1   = vector<double>(nunif, 0.0); // coordinates sampled points (unif. fine grid) 
    xmap2   = vector<double>(nunif, 0.0); // coordinates sampled points (unif. fine grid) 

    // Delta Rij dof
    RkkDelta     = vector<double>(nunif, 0.0);
    thetaZDelta  = vector<double>(nunif, 0.0); 
    thetaYDelta  = vector<double>(nunif, 0.0); 
    thetaXDelta  = vector<double>(nunif, 0.0); 
    xmap1Delta   = vector<double>(nunif, 0.0); 
    xmap2Delta   = vector<double>(nunif, 0.0); 

    // Perturbed & Delta anisotropy tensor dof (in uniform fine grid)
    RxxDelta     = vector<double>(domn->ngrd, 0.0);
    RxyDelta     = vector<double>(domn->ngrd, 0.0);
    RxzDelta     = vector<double>(domn->ngrd, 0.0);
    RyyDelta     = vector<double>(domn->ngrd, 0.0);
    RyzDelta     = vector<double>(domn->ngrd, 0.0);
    RzzDelta     = vector<double>(domn->ngrd, 0.0);
    RxxDeltaUnif = vector<double>(nunif, 0.0);
    RxyDeltaUnif = vector<double>(nunif, 0.0);
    RxzDeltaUnif = vector<double>(nunif, 0.0);
    RyyDeltaUnif = vector<double>(nunif, 0.0);
    RyzDeltaUnif = vector<double>(nunif, 0.0);
    RzzDeltaUnif = vector<double>(nunif, 0.0);

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
    // xmap = lambda_1 * x1c + lambda_2 * x2c + lambda_3 * x3c
    // Realizability Condition (lambda): 0<=lambda_i<=1, sum(lambda_i)=1
    // Realizability Condition (xmap): xmap coord inside barycentric map triangle, defined by x1c, x2c, x3c 
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
    
    // Kronecker delta
    Deltaij       = vector<vector<double>>(3, vector<double>(3, 0.0));
    Deltaij[0][0] = 1.0;
    Deltaij[1][1] = 1.0;
    Deltaij[2][2] = 1.0;

}

void dv_reynolds_stress::updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time) {

    double uvel_fluct, vvel_fluct, wvel_fluct;
    double Rxx_inst, Ryy_inst, Rzz_inst, Rxy_inst, Rxz_inst, Ryz_inst;
    double Rkk_inv;
    double Akk;
    vector<vector<double>> Aij(3, vector<double>(3, 0.0));
    vector<vector<double>> Dij(3, vector<double>(3, 0.0));
    vector<vector<double>> Qij(3, vector<double>(3, 0.0));

    for(int i=0; i<nunif; i++) {

        // ----------------- update reynolds stress tensor -----------------

        uvel_fluct = domn->uvel->dunif.at(i) - domn->uvel->davg.at(i);
        vvel_fluct = domn->vvel->dunif.at(i) - domn->vvel->davg.at(i);
        wvel_fluct = domn->wvel->dunif.at(i) - domn->wvel->davg.at(i);

        Rxx_inst  = pow(uvel_fluct,2);
        Ryy_inst  = pow(vvel_fluct,2);
        Rzz_inst  = pow(wvel_fluct,2);
        Rxy_inst  = uvel_fluct * vvel_fluct;
        Rxz_inst  = uvel_fluct * wvel_fluct;
        Ryz_inst  = vvel_fluct * wvel_fluct;

        Rxx.at(i) = updateTimeMeanQuantity(Rxx_inst, Rxx.at(i), delta_t, averaging_time);
        Ryy.at(i) = updateTimeMeanQuantity(Ryy_inst, Ryy.at(i), delta_t, averaging_time);
        Rzz.at(i) = updateTimeMeanQuantity(Rzz_inst, Rzz.at(i), delta_t, averaging_time);
        Rxy.at(i) = updateTimeMeanQuantity(Rxy_inst, Rxy.at(i), delta_t, averaging_time);
        Rxz.at(i) = updateTimeMeanQuantity(Rxz_inst, Rxz.at(i), delta_t, averaging_time);
        Ryz.at(i) = updateTimeMeanQuantity(Ryz_inst, Ryz.at(i), delta_t, averaging_time);

        // ----------------- update anisotropy tensor -----------------

        // reynolds stress tensor trace (equal to 2*TKE) (1 dof)
        Rkk.at(i) = Rxx.at(i) + Ryy.at(i) + Rzz.at(i);
        Rkk_inv = 1.0 / Rkk.at(i);

        // anisotropy tensor (symmetric, trace free)
        Aij[0][0] = Rkk_inv * Rxx.at(i) - 1.0/3.0;
        Aij[1][1] = Rkk_inv * Ryy.at(i) - 1.0/3.0;
        Aij[2][2] = Rkk_inv * Rzz.at(i) - 1.0/3.0;
        Aij[0][1] = Rkk_inv * Rxy.at(i);
        Aij[0][2] = Rkk_inv * Rxz.at(i);
        Aij[1][2] = Rkk_inv * Ryz.at(i);
        Aij[1][0] = Aij[0][1];
        Aij[2][0] = Aij[0][2];
        Aij[2][1] = Aij[1][2];

        // ensure a_ij is trace-free (previous calc. introduce computational errors)
        Akk = Aij[0][0] + Aij[1][1] + Aij[2][2];
        Aij[0][0] -= Akk/3.0;
        Aij[1][1] -= Akk/3.0;
        Aij[2][2] -= Akk/3.0;

        // ----------------- update eigenvalues of anisotropy tensor  -----------------

        // eigen-decomposition
        domn->eigdec->sym_diagonalize(Aij, Qij, Dij);
        // sort eigenvectors and eigenvalues, with eigenvalues in decreasing order
        domn->eigdec->sortEigenValuesAndEigenVectors(Qij, Dij);

        // Direct barycentric mapping: from eigenvalues matrix to coordinates (2 dof)
        getDirectBarycentricMapping(Dij, xmap1[i], xmap2[i]);  // update xmap, from Dij eigenvalues matrix

        // Rotation angles: from eigenvectors to rotation angles (3 dof)
        getEulerAnglesFromRotationMatrix(Qij, thetaZ[i], thetaY[i], thetaX[i]); // update thetaZ, thetaY, thetaX euler angles, from Qij eigenvectors matrix

    }

}

void dv_reynolds_stress::getReynoldsStressDelta(){

    // perturbed dof of anisotropy tensor
    double RkkPert, thetaZPert, thetaYPert, thetaXPert, xmap1Pert, xmap2Pert; 
    vector<vector<double>> DijPert(3, vector<double>(3, 0.0)); // diag. matrix of eigen-values
    vector<vector<double>> QijPert(3, vector<double>(3, 0.0)); // matrix of eigen-vectors
    vector<vector<double>> RijPert(3, vector<double>(3, 0.0));

    for (int i = 0; i < nunif; i++){

        // perturbed TKE, from RkkDelta
        RkkPert = Rkk[i] + RkkDelta[i];                     // update RkkPert
        enforceRealizabilityRkk(RkkPert);

        // perturbed eigenvalues, from xmapDelta
        xmap1Pert = xmap1[i] + xmap1Delta[i];    
        xmap2Pert = xmap2[i] + xmap2Delta[i];  
        enforceRealizabilityXmap(xmap1Pert, xmap2Pert);                 // update xmap1Pert, xmap2Pert to be realizable, if necessary  
        getInverseBarycentricMapping(xmap1Pert, xmap2Pert, DijPert);    // update DijPert

        // perturbed eigenvectors, from thetaZDelta, thetaYDelta, thetaXDelta
        thetaZPert = thetaZ[i] + thetaZDelta[i];
        thetaYPert = thetaY[i] + thetaYDelta[i];
        thetaXPert = thetaX[i] + thetaXDelta[i]; 
        getRotationMatrixFromEulerAngles(thetaZPert, thetaYPert, thetaXPert, QijPert);  // update QijPert

        // perturbed Rij
        getPerturbedReynoldsStresses(RkkPert, DijPert, QijPert, RijPert); // update RijPert

        // Delta Rij (uniform grid)
        getReynoldsStressesDeltaUnif(RijPert, i);           // update RijDeltaUnif

    }

    // Delta Rij (adaptative grid)
    interpRijDeltaUniformToAdaptativeGrid();                // update RijDelta

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
    Dij[2][2] = - Dij[0][0] - Dij[1][1]; // by constrain: sum(eigenvalues) = 0, source: Emory2013-A
}

/* Calculate rotation angles from rotation matrix of eigenvectors
   Attention: the rotation matrix of eigen-vectors must be indeed a proper rotation matrix. 
   A proper rotation matrix is orthogonal (meaning its inverse is its transpose) and has a determinant of +1.
   This ensures that the matrix represents a rotation without improper reflection or scaling.
   This has been check to be satisfied (+ computational error) at 15 feb. 2024 
*/
void dv_reynolds_stress::getEulerAnglesFromRotationMatrix(const vector<vector<double>> &rotationMatrix, double &thetaZ_i, double &thetaY_i, double &thetaX_i){
    thetaY_i = std::asin(-rotationMatrix[2][0]);
    if (std::cos(thetaY_i) != 0) { // Avoid gimbal lock
        thetaZ_i = std::atan2(rotationMatrix[1][0], rotationMatrix[0][0]);
        thetaX_i = std::atan2(rotationMatrix[2][1], rotationMatrix[2][2]);
    } else {    // Gimbal lock, set yaw to 0 and calculate roll
        thetaZ_i = 0;
        thetaX_i = std::atan2(-rotationMatrix[0][1], rotationMatrix[1][1]);
    }
}


void dv_reynolds_stress::getRotationMatrixFromEulerAngles(const double &thetaZ_i, const double &thetaY_i, const double &thetaX_i, vector<vector<double>> &rotationMatrix){
    // Check if rotationMatrix has the expected shape [3][3]
    if (rotationMatrix.size() != 3 || rotationMatrix[0].size() != 3 || rotationMatrix[1].size() != 3 || rotationMatrix[2].size() != 3) {
        cerr << "Error: rotationMatrix must be of shape [3][3]." << endl;
        return;
    }
    // Calculate trigonometric values
    double cz = cos(thetaZ_i);
    double sz = sin(thetaZ_i);
    double cy = cos(thetaY_i);
    double sy = sin(thetaY_i);
    double cx = cos(thetaX_i);
    double sx = sin(thetaX_i);
    // Calculate the elements of the rotation matrix
    rotationMatrix[0][0] = cy * cz;
    rotationMatrix[0][1] = cy * sz;
    rotationMatrix[0][2] = -sy;
    rotationMatrix[1][0] = (sx * sy * cz) - (cx * sz);
    rotationMatrix[1][1] = (sx * sy * sz) + (cx * cz);
    rotationMatrix[1][2] = sx * cy;
    rotationMatrix[2][0] = (cx * sy * cz) + (sx * sz);
    rotationMatrix[2][1] = (cx * sy * sz) - (sx * cz);
    rotationMatrix[2][2] = cx * cy;
}


void dv_reynolds_stress::getPerturbedReynoldsStresses(const double &RkkPert, const vector<vector<double>> &DijPert, const vector<vector<double>> &QijPert, vector<vector<double>> &RijPert){
    vector<vector<double>> AijPert(3, vector<double>(3, 0.0));
    domn->eigdec->reconstruct_matrix_from_decomposition(DijPert, QijPert, AijPert); // update AijPert
    for (int q = 0; q < 3; q++){
        for (int r = 0; r < 3; r++){
            RijPert[q][r] = RkkPert * ((1.0 / 3.0) * Deltaij[q][r] + AijPert[q][r]);
        }
    }
}


void dv_reynolds_stress::getReynoldsStressesDeltaUnif(const vector<vector<double>> &RijPert, const int &i){
    RxxDeltaUnif[i] = RijPert[0][0] - Rxx[i];
    RxyDeltaUnif[i] = RijPert[0][1] - Rxy[i];
    RxzDeltaUnif[i] = RijPert[0][2] - Rxz[i];
    RyyDeltaUnif[i] = RijPert[1][1] - Ryy[i];
    RyzDeltaUnif[i] = RijPert[1][2] - Ryz[i];
    RzzDeltaUnif[i] = RijPert[2][2] - Rzz[i];
}


void dv_reynolds_stress::interpRijDeltaUniformToAdaptativeGrid(){

    interpVarUnifToAdaptGrid(RxxDeltaUnif, RxxDelta);
    interpVarUnifToAdaptGrid(RxyDeltaUnif, RxyDelta);
    interpVarUnifToAdaptGrid(RxzDeltaUnif, RxzDelta);
    interpVarUnifToAdaptGrid(RyyDeltaUnif, RyyDelta);
    interpVarUnifToAdaptGrid(RyzDeltaUnif, RyzDelta);
    interpVarUnifToAdaptGrid(RzzDeltaUnif, RzzDelta);

}


///////////////////////////  Enforce realizability of perturbed xmap  ///////////////////////////

/* Transform cartesian coordinates 'xmapping' to barycentric coordinates 'lambda' of eigenvalues barycentric map
        xmapping1,2: cartesian coordinates of barycentric map
        lambda1,2,3: barycentric coordinates of barycentric map, satifying lambda1+lambda2+lambda3=1
        transformation:  lambda1,2 = Tinv * (xmapping_1,2 - t) 
 */
void dv_reynolds_stress::getBarycentricCoordFromCartesianCoord(const double &xmapping1, const double &xmapping2, vector<double> &lambda){
    // Assuming lambda is always of size 3
    vector<double> xmapping = {xmapping1, xmapping2};
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            lambda[i] += Tinv[i][j] * (xmapping[j] - t[j]);
        }
    }
    lambda[2] = 1.0 - lambda[0] - lambda[1];    // from barycentric coord. condition sum(lambda_i) = 1.0
}

/* Transform barycentric coordinates 'lambda' to cartesian coordinates 'xmapping' of eigenvalues barycentric map
        transformation: xmapping = lambda_1 * x1c + lambda_2 * x2c + lambda_3 * x3c
 */
void dv_reynolds_stress::getCartesianCoordFromBarycentricCoord(const vector<double> &lambda, double &xmapping1, double &xmapping2){
    // Assuming lambda is always of size 3
    xmapping1 = lambda[0] * x1c[0] + lambda[1] * x2c[0] + lambda[2] * x3c[0];
    xmapping2 = lambda[0] * x1c[1] + lambda[1] * x2c[1] + lambda[2] * x3c[1];
}

/* Check all elements of barycentric coordinates are in range [0.0, 1.0]
        return: true if all elements are contained within [0.0, 1.0], false otherwise
 */
bool dv_reynolds_stress::areRealizableBarycentricCoord(const vector<double> &lambda) {
    // Assuming lambda is always of size 3
    return 0.0 <= lambda[0] && lambda[0] <= 1.0 &&
           0.0 <= lambda[1] && lambda[1] <= 1.0 &&
           0.0 <= lambda[2] && lambda[2] <= 1.0;
}

/* Modifies barycentric coordinates of eigenvalues barycentric map to be realizable, by:
        1st: truncate each coordinate to be within [0,1] range
        2nd: normalize the coordinates vector to have the sum of elements = 1
 */
void dv_reynolds_stress::truncateAndNormalizeBarycentricCoord(vector<double> &lambda){
    // Assuming lambda is always of size 3
    // 1st: truncate coordinates within range [0,1]
    for (int i = 0; i < 3; i++)
        lambda[i] = std::min(std::max(lambda[i], 0.0), 1.0); 
    // 2nd: normalize coordinates vector to sum(coordinates) = 1
    double sumCoord = lambda[0] + lambda[1] + lambda[2];
    for (int i = 0; i < 3; i++)
        lambda[i] /= sumCoord; 
}

// Check realizability of Rkk (Rij norm), by garanteeing Rkk >= 0
void dv_reynolds_stress::enforceRealizabilityRkk(double &rkk){
    if (rkk<0.0){
        rkk=0.0;
    }
}

/* Check realizability of xmap coordinates (of eigenvalues) by:
   1st. transforming xmap cartesian coordinates to barycentric coordinates
   2nd. xmapping is inside the barycentric map triangle (defined by x1c, x2c, x3c, thus realizable) 
        iff lambda_i>=0 && sum_i(lambda_i)=1
   Attention: sum_i(lambda_i)=1 condition is already garanteed by construction in 'getBarycentricCoordFromCartesianCoord' method,
              only 0<=lambda_i<=1 condition is assessed.
*/
void dv_reynolds_stress::enforceRealizabilityXmap(double &xmap1, double &xmap2){
    vector<double> lambda(3, 0.0);
    getBarycentricCoordFromCartesianCoord(xmap1, xmap2, lambda);    // update lambda
    // Check eigenvalues realizability from barycentric coordinates:
    if (!areRealizableBarycentricCoord(lambda)){ 
        // enforce realizability by truncating & normalizing barycentric coordinates 'lambda'
        // cout << "\nPerturbed xmap not realizable - outside barycentric map realizability triangle." << endl;
        // cout << "Non-realizable barycentric coordinates lambda = (" << lambda[0] << ", " << lambda[1] << ", " << lambda[2] << ")" << endl;
        truncateAndNormalizeBarycentricCoord(lambda);               // update lambda
        // cout << "Realizable barycentric coordinates lambda = (" << lambda[0] << ", " << lambda[1] << ", " << lambda[2] << ")" << endl;
    }
    getCartesianCoordFromBarycentricCoord(lambda, xmap1, xmap2);    // update xmap1, xmap2
}