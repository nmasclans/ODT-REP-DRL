/**
 * @file dv_reynolds_stress.cc
 * @brief Source file for class dv_reynolds_stress
 */

#include "dv_reynolds_stress.h"
#include "domain.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

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

    // todo: change variable names eigVal eigVect to diagDij and Qij everywhere for consistency
    // anisotropy tensor
    // -> trace = 2*TKE
    Rkk     = vector<double>(nunif, 0.0);
    // -> eigenvalues, shape [nunif, neig] = [nunif, 3]
    eigVal  = vector<vector<double>>(nunif, vector<double>(3, 0.0));
    // -> eigenvectors, shape [nunif, neig, ndim] = [nunif, 3, 3]
    eigVect = vector<vector<vector<double>>>(nunif, vector<vector<double>>(3, vector<double>(3, 0.0)));

    // barycentric map - coordinates (2 dof)
    x1c     = vector<double>{1.0, 0.0};                               // corner x1c
    x2c     = vector<double>{0.0, 0.0};                               // corner x2c
    x3c     = vector<double>{0.5, sqrt(3.0) / 2.0};                   // corner x3c
    xmap    = vector<vector<double>>(nunif, vector<double>(2, 0.0)); // coordinates sampled points (unif. fine grid)

    // rotation angles (3 dof)
    thetaZ  = vector<double>(nunif, 0.0); 
    thetaY  = vector<double>(nunif, 0.0); 
    thetaX  = vector<double>(nunif, 0.0); 

    // Delta Rij dof
    RkkDelta     = vector<double>(nunif, 0.0);
    thetaZDelta  = vector<double>(nunif, 0.0); 
    thetaYDelta  = vector<double>(nunif, 0.0); 
    thetaXDelta  = vector<double>(nunif, 0.0); 
    xmapDelta    = vector<vector<double>>(nunif, vector<double>(2, 0.0)); 

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
        // store eigenvectors and eigenvalues
        for (int q=0; q<3; q++){
            eigVal[i][q] = Dij[q][q];
            for (int r = 0; r < 3; r++){
                eigVect[i][q][r] = Qij[q][r];
            }
        }

        // Direct barycentric mapping: from eigenvalues to coordinates (2 dof)
        getDirectBarycentricMapping(eigVal[i], xmap[i]);

        // Rotation angles: from eigenvectors to rotation angles (3 dof)
        getEulerAnglesFromRotationMatrix(eigVect[i], thetaZ[i], thetaY[i], thetaX[i]);

    }

}

void dv_reynolds_stress::getReynoldsStressDelta(){

    // perturbed dof of anisotropy tensor
    double RkkPert, thetaZPert, thetaYPert, thetaXPert;
    vector<double> xmapPert(2, 0.0); 
    vector<double> eigValPert(3, 0.0);
    vector<vector<double>> DijPert(3, vector<double>(3, 0.0)); // diag. matrix of eigen-values
    vector<vector<double>> QijPert(3, vector<double>(3, 0.0)); // matrix of eigen-vectors
    vector<vector<double>> RijPert(3, vector<double>(3, 0.0));

    for (int i = 0; i < nunif; i++){

        // perturbed TKE, from RkkDelta
        RkkPert = Rkk[i] + RkkDelta[i];                     // update RkkPert

        // perturbed eigenvalues, from xmapDelta
        for (int j = 0; j < 2; j++)
            xmapPert[j] = xmap[i][j] + xmapDelta[i][j];    
        getInverseBarycentricMapping(xmapPert, eigValPert);
        for (int q = 0; q < 3; q++)
            DijPert[q][q] = eigValPert[q];                  // update DijPert

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
void dv_reynolds_stress::getDirectBarycentricMapping(const vector<double> &eigenvalues, vector<double> &xmapping){
    for (int i=0; i<2; i++) {
        xmapping[i] = x1c[i] * (    eigenvalues[0] - eigenvalues[1]) \
                    + x2c[i] * (2.0*eigenvalues[1] - 2.0*eigenvalues[2]) \
                    + x3c[i] * (3.0*eigenvalues[2] + 1.0);
    }
}


// Inverse barycentric mapping: from barycentric coordinates to eigenvalues
void dv_reynolds_stress::getInverseBarycentricMapping(const vector<double> &xmapping, vector<double> &eigenvalues){
    for (int i = 0; i < 2; i++){
        eigenvalues[i] = 0.0;
        for (int j = 0; j < 2; j++){
            eigenvalues[i] += Binv[i][j] * (xmapping[j] - b[j]);
        }
    }
    eigenvalues[2] = - eigenvalues[0] - eigenvalues[1]; // by constrain: sum(eigenvalues) = 0
}


// Calculate rotation angles from rotation matrix of eigenvectors
/* Attention: the rotation matrix of eigen-vectors must be indeed a proper rotation matrix. 
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
    domn->eigdec->reconstruct_matrix_from_decomposition(DijPert, QijPert, AijPert); // updates AijPert
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