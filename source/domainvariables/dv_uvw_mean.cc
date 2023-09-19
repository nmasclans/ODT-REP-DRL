/**
 * @file dv_uvw_mean.cc
 * @brief Source file for class dv_uvw_mean
 */

// todo: eliminate subclass if considered

#include "dv_uvw_mean.h"
#include "domain.h"
#include <cstdlib>
#include <cmath>


////////////////////////////////////////////////////////////////////////////////
/*! dv_uvw_mean  constructor function
 *
 * @param p_domn  \input set domain pointer with.
 * @param p_phi   \input set vector pointer with.
 */

dv_uvw_mean::dv_uvw_mean(domain    *line,
                         const      string s,
                         const bool Lt,
                         const bool Lo) {

    domn          = line;
    var_name      = s;
    L_transported = Lt;
    L_output      = Lo;
    L_output_stat = false;
    d             = vector<double>(domn->ngrd, 0.0);
    tLastAvg      = 0.0;
    tBeginAvg     = domn->pram->tBeginAvg;
    counter       = 0; // TODO: erase counter

    // corresponding instantaneous value name for the mean velocity component <var_name>
    if (var_name == "uvelmean") {var_name_inst = "uvel";}
    else if (var_name == "vvelmean") {var_name_inst = "vvel";}
    else if (var_name == "wvelmean") {var_name_inst = "wvel";}
    else { 
        cout << endl << "ERROR in dv_uvw_mean::updateStatistics, invalid var_name = " << var_name << ", accepted values: uvelmean, vvelmean, wvelmean." << endl; 
        exit(0);
    }

}

////////////////////////////////////////////////////////////////////////////////
/*! Set uvw mean value from the mean value of previous time step and the 
 *  current instantaneous value
 *  @param ipt \input optional point to compute at
 */

// void dv_uvw_mean::setVar(const int ipt){
// 
//     if(ipt != -1) {
//         cout << endl << "ERROR in setVar: ipt must be -1" << endl;
//         exit(0);
//     }
//     if(domn->pram->Lspatial or domn->pram->cCoord != 1){
//         cout << endl << "ERROR in setVar: dv_uvw_mean::setVar not implemented for Lspatial = true or cCoord != 1.0" << endl;
//         exit(0);
//     }
// 
// }

void dv_uvw_mean::updateStatisticsIfNeeded(const double &time) {
    // TODO: ERASE ALL INSTANCES OF COUNTER
    double tAvg;
    double dt;
    
    if (time > tBeginAvg){ 
        // calculate averaging time and delta time
        tAvg = time - tBeginAvg;
        dt   = tAvg - tLastAvg;
        // update statistic
        for(int k=0; k<domn->v.size(); k++){
            if(domn->v.at(k)->var_name == var_name_inst){
                if (counter < 500 && var_name_inst == "uvel") {
                    cout << scientific; 
                    cout << "\nFor var_name = " << var_name << "  //  var_name_inst = " << var_name_inst << endl;
                    cout << "tAvg = " << tAvg << ", tLastAvg = " << tLastAvg << ", dt = " << dt << endl;  
                    cout << "size velmean = " << d.size() << "  //  size vel = " << domn->v.at(k)->d.size() << "  //  size pos = " << domn->pos->d.size() << endl;
                    cout << "velmean_old(5) = " << d.at(5) << "  //  vel(5) = " << domn->v.at(k)->d.at(5) << "  //  pos(5) = " << domn->pos->d.at(5) << endl;
                }
                for(int i=0; i<d.size(); i++) {
                    d.at(i) = ( tLastAvg * d.at(i) + dt * domn->v.at(k)->d.at(i) ) / tAvg;
                }
                if (counter < 500 && var_name_inst == "uvel") {
                    cout << "velmean(5) = " << d.at(5) << endl;
                }
            }
        } 
        // update time quantities
        tLastAvg    = tAvg;
        counter    += 1;
    } else {
        d.resize(domn->ngrd,0.0);
    }

}