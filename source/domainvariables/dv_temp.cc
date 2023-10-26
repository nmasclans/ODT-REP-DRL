/**
 * @file dv_temp.cc
 * @brief Source file for class dv_temp
 */


#include "dv_temp.h"
#include "domain.h"
#include <cstdlib>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
/*! dv_temp  constructor function
 *
 * @param p_domn  \input set domain pointer with.
 * @param p_phi   \input set vector pointer with.
 */

// todo: change this constructor and all constructor of dv_ subclasses to construct the parent class dv (as done in dv_uvw)
// this would lead to eliminate all the superfluous definitions of domn, var_name, L_... which are already defined in the dv parent class
dv_temp::dv_temp(domain  *line,
                 const      string s,
                 const bool Lt,
                 const bool Lo) {

    domn          = line;
    var_name      = s;
    L_transported = Lt;
    L_output      = Lo;
    L_output_stat = false;
    d             = vector<double>(domn->ngrd, 0.0);

}

////////////////////////////////////////////////////////////////////////////////
/*! Set temperature from the gas state
 *  @param ipt \input optional point to compute at
 */

void dv_temp::setVar(const int ipt){

    d.resize(domn->ngrd);
    if(ipt == -1)
        for(int i=0; i<domn->ngrd; i++) {
            domn->domc->setGasStateAtPt(i);
            d.at(i) = domn->gas->temperature();
        }
    else {
        domn->domc->setGasStateAtPt(ipt);
        d.at(ipt) = domn->gas->temperature();
    }
}

