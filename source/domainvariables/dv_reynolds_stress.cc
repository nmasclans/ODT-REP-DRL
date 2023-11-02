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

    // num. points uniform fine grid
    nunif = domn->pram->nunif;

    // Reynolds stress terms
    Rxx     = vector<double>(nunif, 0.0); 
    Ryy     = vector<double>(nunif, 0.0); 
    Rzz     = vector<double>(nunif, 0.0); 
    Rxy     = vector<double>(nunif, 0.0); 
    Rxz     = vector<double>(nunif, 0.0); 
    Ryz     = vector<double>(nunif, 0.0);

    // anisotropy tensor - eigenvalues
    lambda0 = vector<double>(nunif, 0.0);
    lambda1 = vector<double>(nunif, 0.0);
    lambda2 = vector<double>(nunif, 0.0);

    // barycentric map - corners
    x1c     = vector<double> {1.0, 0.0};
    x2c     = vector<double> {0.0, 0.0};
    x3c     = vector<double> {0.5, sqrt(3.0)/2.0};
    xmap    = vector<vector<double>>(nunif, vector<double>(2, 0.0));

}

void dv_reynolds_stress::updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time) {

    double Rxx_inst, Ryy_inst, Rzz_inst, Rxy_inst, Rxz_inst, Ryz_inst;
    double Rkk_inv;
    double Aij[3][3] = {};
    double Dij[3][3] = {};
    double Qij[3][3] = {};

    for(int i=0; i<nunif; i++) {

        // ----------------- update reynolds stress tensor -----------------
        
        Rxx_inst = pow(domn->uvel->drmsf.at(i),2);
        Ryy_inst = pow(domn->vvel->drmsf.at(i),2);
        Rzz_inst = pow(domn->wvel->drmsf.at(i),2);
        Rxy_inst = domn->uvel->drmsf.at(i) * domn->vvel->drmsf.at(i);
        Rxz_inst = domn->uvel->drmsf.at(i) * domn->wvel->drmsf.at(i);
        Ryz_inst = domn->vvel->drmsf.at(i) * domn->wvel->drmsf.at(i);
        
        Rxx.at(i) = updateTimeMeanQuantity(Rxx_inst, Rxx.at(i), delta_t, averaging_time);
        Ryy.at(i) = updateTimeMeanQuantity(Ryy_inst, Ryy.at(i), delta_t, averaging_time);
        Rzz.at(i) = updateTimeMeanQuantity(Rzz_inst, Rzz.at(i), delta_t, averaging_time);
        Rxy.at(i) = updateTimeMeanQuantity(Rxy_inst, Rxy.at(i), delta_t, averaging_time);
        Rxz.at(i) = updateTimeMeanQuantity(Rxz_inst, Rxz.at(i), delta_t, averaging_time);
        Ryz.at(i) = updateTimeMeanQuantity(Ryz_inst, Ryz.at(i), delta_t, averaging_time);
    
        // ----------------- update anisotropy tensor -----------------

        // reynolds stress tensor trace (equal to 2*TKE)
        Rkk_inv = 1.0 / ( Rxx.at(i) + Ryy.at(i) + Rzz.at(i) );

        // anisotropy tensor (symmetric, trace free)
        Aij[0][0] = Rkk_inv * Rxx.at(i) - 1.0/3.0;
        Aij[1][1] = Rkk_inv * Ryy.at(i) - 1.0/3.0;
        Aij[2][2] = Rkk_inv * Rzz.at(i) - 1.0/3.0;
        Aij[0][1] = Rkk_inv * Rxy.at(i);                Aij[1][0] = Aij[0][1];
        Aij[0][2] = Rkk_inv * Rxz.at(i);                Aij[2][0] = Aij[0][2];
        Aij[1][2] = Rkk_inv * Ryz.at(i);                Aij[2][1] = Aij[1][2];

        // ----------------- update eigenvalues of anisotropy tensor  -----------------

        domn->eigdec->sym_diagonalize(Aij, Qij, Dij);

        // eigenvalues, with lambda0 >= lambda1 >= lambda2
        lambda0.at(i) = Dij[0][0];
        lambda1.at(i) = Dij[1][1];
        lambda2.at(i) = Dij[2][2];
        
        // barycentric map
        xmap[i][0] =   x1c.at(0) * (    lambda0.at(i) - lambda1.at(i)) \
                     + x2c.at(0) * (2.0*lambda1.at(i) - 2.0*lambda2.at(i)) \
                     + x3c.at(0) * (3.0*lambda2.at(i) + 1.0);
        xmap[i][1] =   x1c.at(1) * (    lambda0.at(i) - lambda1.at(i)) \
                     + x2c.at(1) * (2.0*lambda1.at(i) - 2.0*lambda2.at(i)) \
                     + x3c.at(1) * (3.0*lambda2.at(i) + 1.0);

    }

}
