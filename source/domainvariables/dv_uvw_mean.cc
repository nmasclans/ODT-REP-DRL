/**
 * @file dv_uvw_mean.cc
 * @brief Source file for class dv_uvw_mean
 */


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
    d             = vector<double>(domn->ngrd, 0.0);
    tLast         = 0.0;

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

void dv_uvw_mean::setVar(const int ipt){

    if(ipt != -1) {
        cout << endl << "ERROR in setVar: ipt must be -1" << endl;
        exit(0);
    }
    if(domn->pram->Lspatial or domn->pram->cCoord != 1){
        cout << endl << "ERROR in setVar: dv_uvw_mean::setVar not implemented for Lspatial = true or cCoord != 1.0" << endl;
        exit(0);
    }

}


void dv_uvw_mean::update(const double &time) {

    for(int k=0; k<domn->v.size(); k++){
        if(domn->v.at(k)->var_name == var_name_inst){

            // TODO: fix problem of 0.0 vectors!
            cout << scientific; 
            cout << "(dv_uvw_mean::update) For var_name = " << var_name << "  //  var_name_inst = " << var_name_inst << endl;
            for(int i=0; i<d.size(); i++) {
                cout << "velmean(" << i << ") = " << d.at(i) << "  //  vel(" << i << ") = " << domn->v.at(k)->d.at(i) << endl;
                d.at(i) = ( tLast * d.at(i) + ( time - tLast ) * domn->v.at(k)->d.at(i)) / time;
            }
        }
    } 
    cout << "\n\n";

    // update time quantities
    tLast    = time;

}