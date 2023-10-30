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
    Rxx   = vector<double>(nunif, 0.0); 
    Ryy   = vector<double>(nunif, 0.0); 
    Rzz   = vector<double>(nunif, 0.0); 
    Rxy   = vector<double>(nunif, 0.0); 
    Rxz   = vector<double>(nunif, 0.0); 
    Ryz   = vector<double>(nunif, 0.0); 

}

void dv_reynolds_stress::updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time) {

    double Rxx_inst, Ryy_inst, Rzz_inst, Rxy_inst, Rxz_inst, Ryz_inst;

    for(int i=0; i<nunif; i++) {
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
    }

}