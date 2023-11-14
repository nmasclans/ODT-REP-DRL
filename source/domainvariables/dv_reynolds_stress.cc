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

    // Parameters
    nunif          = domn->pram->nunif;
    factEigValPert = domn->pram->factEigValPert;
    xmapTarget     = vector<double>{domn->pram->xmapTarget1,domn->pram->xmapTarget2};

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
    // -> eigenvalues, shape [nunif, neig] = [nunif, 3]
    eigVal  = vector<vector<double>>(nunif, vector<double>(3, 0.0));
    // -> eigenvectors, shape [nunif, neig, ndim] = [nunif, 3, 3]
    eigVect = vector<vector<vector<double>>>(nunif, vector<vector<double>>(3, vector<double>(3, 0.0)));

    // barycentric map - coordinates
    x1c     = vector<double>{1.0, 0.0};                               // corner x1c
    x2c     = vector<double>{0.0, 0.0};                               // corner x2c
    x3c     = vector<double>{0.5, sqrt(3.0) / 2.0};                   // corner x3c
    xmap    = vector<vector<double>>(nunif, vector<double>(2, 0.0)); // coordinates sampled points (unif. fine grid)

    // Perturbed & Delta anisotropy tensor dof
    RijDelta = vector<vector<vector<double>>>(nunif, vector<vector<double>>(3, vector<double>(3, 0.0)));

    // Direct mapping:  xmap = B * eigenvalues + b, where b = x3c
    b = vector<double>(2, 0.0);
    vector<vector<double>> B(2, vector<double>(2, 0.0));
    for (int i=0; i<2; i++){
        b[i]     = x3c[i];
        B[i][0]  = x1c[i] + 2 * x2c[i] - 3 * x3c[i];
        B[i][1]  = -x1c[i] + 4 * x2c[i] - 3 * x3c[i];
    }

    // Inverse mapping: eigenvalues = B^{-1} * (xmap - b), where B^{-1}:
    double Bdet  = B[0][0] * B[1][1] - B[0][1] * B[1][0];
    Binv         = vector<vector<double>>(2, vector<double>(2, 0.0));
    Binv[0][0]   = B[1][1] / Bdet;
    Binv[0][1]   = -B[0][1] / Bdet;
    Binv[1][0]   = -B[1][0] / Bdet;
    Binv[1][1]   = B[0][0] / Bdet;

    // Kronecker delta
    Deltaij      = vector<vector<double>>(3, vector<double>(3, 0.0));

    // Target barycentric map coordinates and eigenvalues
    eigValTarget = vector<double>(3, 0.0);
    getInverseBarycentricMapping(xmapTarget, eigValTarget);
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

        // reynolds stress tensor trace (equal to 2*TKE)
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

        domn->eigdec->sym_diagonalize(Aij, Qij, Dij);

        // store eigenvectors and eigenvalues
        for (int q=0; q<3; q++){
            eigVal[i][q] = Dij[q][q];
            for (int r = 0; r < 3; r++){
                eigVect[i][q][r] = Qij[q][r];
            }
        }

        // Direct barycentric mapping: from eigenvalues to coordinates
        getDirectBarycentricMapping(eigVal[i], xmap[i]);
        /// for (int q=0; q<2; q++){
        ///     xmap[i][q] =   x1c[q] * (    eigVal[i][0] -     eigVal[i][1]) \
        ///                  + x2c[q] * (2.0*eigVal[i][1] - 2.0*eigVal[i][2]) \
        ///                  + x3c[q] * (3.0*eigVal[i][2] + 1.0);
        /// }

    }

}

void dv_reynolds_stress::getReynoldsStressDelta(){

    // perturbed dof of anisotropy tensor
    double RkkPert = 0.0;
    vector<vector<double>> DijPert(3, vector<double>(3, 0.0)); // diag. matrix of eigen-values
    vector<vector<double>> QijPert(3, vector<double>(3, 0.0)); // matrix of eigen-vectors
    vector<vector<double>> RijPert(3, vector<double>(3, 0.0));

    for (int i = 0; i < nunif; i++){

        // perturbed TKE - not implemented, same as current
        getPerturbedTrace(Rkk[i], RkkPert); // update RkkPert

        // perturbed eigenvalues - implemented
        getPerturbedEigenValuesMatrix(eigVal[i], DijPert); // updates eigValPert

        // perturbed eigenvectors - not implemented, same as current
        getPerturbedEigenVectorsMatrix(eigVect[i], QijPert); // update eigVectPert

        // perturbed Rij
        getPerturbedReynoldsStresses(RkkPert, DijPert, QijPert, RijPert); // update RijPert

        // Delta Rij
        getRijDelta(RijPert, i); // update RijDelta

    }
}

void dv_reynolds_stress::getPerturbedTrace(const double &Rkk, double &RkkPert){
    RkkPert = Rkk;
}

void dv_reynolds_stress::getPerturbedEigenValuesMatrix(const vector<double> &eigVal, vector<vector<double>> &DijPert){
    for (int q = 0; q < 3; q++){
        DijPert[q][q] = (1 - factEigValPert) * eigVal[q] + factEigValPert * eigValTarget[q];
    }
}

void dv_reynolds_stress::getPerturbedEigenVectorsMatrix(const vector<vector<double>> &eigVect, vector<vector<double>> &QijPert){
    for (int q = 0; q < 3; q++){
        for (int r = 0; r < 3; r++){
            QijPert[q][r] = eigVect[q][r];
        }
    }
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

void dv_reynolds_stress::getRijDelta(const vector<vector<double>> &RijPert, const int &i){
    RijDelta[i][0][0] = RijPert[0][0] - Rxx[i];
    RijDelta[i][0][1] = RijPert[0][1] - Rxy[i];
    RijDelta[i][0][2] = RijPert[0][2] - Rxz[i];
    RijDelta[i][1][1] = RijPert[1][1] - Ryy[i];
    RijDelta[i][1][2] = RijPert[1][2] - Ryz[i];
    RijDelta[i][2][2] = RijPert[2][2] - Rzz[i];
    RijDelta[i][1][0] = RijDelta[i][0][1];
    RijDelta[i][2][0] = RijDelta[i][0][2];
    RijDelta[i][2][1] = RijDelta[i][1][2];
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
    eigenvalues[2] = -eigenvalues[0] - eigenvalues[1]; // by constrain: sum(eigenvalues) = 0
}