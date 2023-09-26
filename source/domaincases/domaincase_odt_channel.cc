/**
 * @file domaincase_odt_channel.cc
 * @brief Source file for class domaincase_odt_channel
 */

#include "domaincase_odt_channel.h"
#include "domain.h"
#include "dv.h"
#include "dv_pos.h"
#include "dv_posf.h"
#include "dv_rho_const.h"
#include "dv_dvisc_const.h"
#include "dv_uvw.h"

////////////////////////////////////////////////////////////////////////////////
/** Initialization
 *
 * @param p_domn  \input set domain pointer with.
 * @param p_phi   \input set vector pointer with.
 */

void domaincase_odt_channel::init(domain *p_domn){

    // store pointer of the domain object
    domn = p_domn; 

    // add different variable types (being objects of dv subclasses) to the 'v' vector, class member of the 'domain' object
    // each of these variables is created using their respective constr. and is pushed into the 'v' vector<dv*> of 'domain' object
    domn->v.push_back(new dv_pos(        domn, "pos",      false, true));   // last are: L_transported, L_output
    domn->v.push_back(new dv_posf(       domn, "posf",     false, true));
    domn->v.push_back(new dv_rho_const(  domn, "rho",      false, false));
    domn->v.push_back(new dv_dvisc_const(domn, "dvisc",    false, false));
    domn->v.push_back(new dv_uvw(        domn, "uvel",     true,  true, domn->pram->Lstatconv));
    domn->v.push_back(new dv_uvw(        domn, "vvel",     true,  true));
    domn->v.push_back(new dv_uvw(        domn, "wvel",     true,  true));

    // assign specific elements from the 'v' vector to various member variables of the 'domain' object
    int j = 0;
    domn->pos      = domn->v.at(j++);
    domn->posf     = domn->v.at(j++);
    domn->rho      = domn->v.at(j++);
    domn->dvisc    = domn->v.at(j++);
    domn->uvel     = domn->v.at(j++);
    domn->vvel     = domn->v.at(j++);
    domn->wvel     = domn->v.at(j++);

    //------------------- set variables used for mesh adaption

    // a vector of pointers to 'dv' objects is created and initialized with the 'uvel' variable
    // the domain member variable 'mesher' (of class 'meshManager') is initialized 
    vector<dv*> phi;
    phi.push_back(domn->uvel);
    domn->mesher->init(domn, phi);

    //------------------- set inlet_cell_dv_props for inlet cell inserted for suction/blowing case

    // the 'domaincase' class has data member 'inlet_cell_dv_props', a vector that lists all dv properties
    // for inserted inlet cell for channel suction/blowing case 
    // channelFlow: the stored properties corresponts to the 1D odt line bottom, for y-coord minim 
    inlet_cell_dv_props.resize(domn->v.size());

    inlet_cell_dv_props[0] = -1;                                     // pos:  set elsewhere
    inlet_cell_dv_props[1] = -domn->pram->domainLength/2.0;          // posf:
    inlet_cell_dv_props[2] = domn->pram->rho0;                       // rho:
    inlet_cell_dv_props[3] = domn->pram->kvisc0 * domn->pram->rho0;  // dvisc:
    inlet_cell_dv_props[4] = domn->pram->uBClo;                      // uvel:
    inlet_cell_dv_props[5] = domn->pram->vBClo;                      // vvel:
    inlet_cell_dv_props[6] = domn->pram->wBClo;                      // wvel:


    //------------------- default velocity values (0.0) are fine, along with rho, dvisc.

    //for(int i=0; i<domn->uvel->d.size(); i++)
    //  domn->uvel->d[i] = 10*domn->pos->d.at(i);
    //  domn->uvel->d[i] = 10*0.016/4.0/0.002*(1.0-domn->pos->d.at(i)*domn->pos->d.at(i));

}

////////////////////////////////////////////////////////////////////////////////
/** Update/set variables that are needed in the soltuion.
 *  Especially for diffusion.
 *  These should not be transported. Those are already set.
 */

void domaincase_odt_channel::setCaseSpecificVars() {

    domn->rho->setVar();
    domn->dvisc->setVar();

}
