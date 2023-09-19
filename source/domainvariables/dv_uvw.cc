/**
 * @file dv_uvw.cc
 * @brief Source file for class dv_uvw
 */


#include "dv_uvw.h"
#include "domain.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "interp_linear.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! dv_uvw  constructor function
 *
 * @param p_domn  \input set domain pointer with.
 * @param p_phi   \input set vector pointer with.
 * channelFlow: sets L_transported = true, L_output = true
 */

dv_uvw::dv_uvw(domain  *line,
               const      string s,
               const bool Lt,
               const bool Lo,
               const bool Lsc) {
    
    domn          = line;
    var_name      = s;
    L_transported = Lt;
    L_output      = Lo;
    d             = vector<double>(domn->ngrd, 0.0); // variable value along at the grid points of the domain 
    
    // -> N-S Eq data members 
    rhsSrc        = vector<double>(domn->ngrd, 0.0);
    rhsMix        = vector<double>(domn->ngrd, 0.0);
    rhsStatConv   = vector<double>(domn->ngrd, 0.0);
    
    // -> Statistics data members
    L_output_stat = true;   // todo: add this as a domn->pram, include it into the input.yaml file
    davg          = vector<double>(domn->ngrd, 0.0); // todo: add description
    posLast       = vector<double>(domn->ngrd, 0.0); // todo: add description
    tLastAvg      = 0.0;
    tBeginAvg     = domn->pram->tBeginAvg;
    // corresponding instantaneous value name for the mean velocity component <var_name>
    if (var_name == "uvel") {var_name_stat = "uvelmean";}
    else if (var_name == "vvel") {var_name_stat = "vvelmean";}
    else if (var_name == "wvel") {var_name_stat = "wvelmean";}
    else {cout << endl << "ERROR in dv_uvw initialization, invalid var_name = " << var_name << ", accepted values: uvel, vvel, wvel." << endl; exit(0); }

    // -> Statistics convergence
    L_statconv    = Lsc;
    dvaldt        = vector<double>(domn->ngrd, 0.0); // variable discrete time derivative along at the grid points of the domain 
    // todo: what to do with all the framework for calculating dvaldt, and initializing dvaldt in various .cc and .h files?

}

////////////////////////////////////////////////////////////////////////////////
/*! lv source term part of the rhs function. 
 *  Method implementation for source term of the right-hand side (Rhs) 
 *  @param ipt \input optional point to compute source at.
 */

void dv_uvw::getRhsSrc(const int ipt){

    /* if L_transported = False then no calculation of source terms in Rhs is done
     * L_transported = False means that the variable represented by the instance of 'dv_uvw' is not being transported, 
     * (i.e., no source term is needed). In that case, the method returns early, and no further calculations 
     * are performed
     * channelFlow: sets L_transported = True for variables 'uvel', 'vvel', and 'wvel' (all of class dv_uvw)
     */
    if(!L_transported)
        return;

    /* The 'rhsSrc' vector is resized to match the size of the grid in the associated 'domain' object ('domn').
     * It is initialized with zeros.
     * The 'rhsSrc' vector will eventually store the calculated source terms (Src) in the right-hand side (rhs)  
     */
    rhsSrc.resize(domn->ngrd, 0.0);

    //-------------------------
    // Calculation of source terms based on conditions

    /* if ipt==-1, the source term calculations will be applied to all grid points
     * this is done because the ipt comes from the domain index of a certain point, with index -1 refering
     * to a point grid outside the grid's domain, done so that the source term is applied in all domain points.
     * Therefore, the source terms with ipt==-1 indicate that the source term calculation couldn't be performed
     * for a specific grid point because the given position does not map to any grid point in the domain, but
     * they have to be applied everywhere.   
     * channelFlow: ipt==-1 is satisfied
     */ 
    if(ipt==-1) {
        if(var_name == "uvel" && domn->pram->cCoord != 3.0) {
            /* The 'rhsSrc' vector is assigned a new vector with a size 'domn->ngrid', filled
             * with the value '-domn->pram->dPdx'  (-dP/dx) for the 'uvel' variable,
             * when 'domn->pram->cCoord != 3.0' (non-spherical coordinates).
             * input param cCoord referers to the coordinate type, being 1 = planar, 2 = cylindrical, 3 = spherical
             * channelFlow: cCoord is set to 1 in the input.yaml file.
             */
            // pressure gradient source term
            rhsSrc = vector<double>(domn->ngrd, -domn->pram->dPdx);

            /* If buoyancy is enabled (domn->pram->Lbuoyant = True), this loop iterates over
             * the grid points and adjusts the values of the 'rhsSrc' vector.
             * A contribution to the 'rhsSrc' is added based on the density difference between the 
             * current point and the last point in the grid, and divides the result by the density
             * at the current point.
             */
            // add buoyancy source term
            // channelFlow: Lbuoyant = false by default, no buoyancy term added
            for(int i=0; i<domn->ngrd; i++) {
                if(domn->pram->Lbuoyant)
                    rhsSrc.at(i) += (domn->rho->d.at(i) - domn->rho->d.at(domn->ngrd-1))*domn->pram->g;
                rhsSrc.at(i) /= domn->rho->d.at(i);
            }
        }

        /* Parameter domn->pram->Lspatial = False by default, if not specified differently in the input.yaml
         * channelFlow: Lspatial = False, as ODT is 1D, and solves in time
         * other: Lspatial may be True for ODT solving in 2D, with no time advancement  
         */
        if(domn->pram->Lspatial)
            for(int i=0; i<domn->ngrd; i++)
                rhsSrc.at(i) /= domn->uvel->d.at(i);
    }
    // if ipt != -1 indicates the source term must be applied in a particular point index inside the domain,
    // not the case for channel flow
    else {
        if(var_name == "uvel" && domn->pram->cCoord != 3.0) {
            rhsSrc.at(ipt) = -domn->pram->dPdx;
            if(domn->pram->Lbuoyant)
                rhsSrc.at(ipt) += (domn->rho->d.at(ipt) - domn->rho->d.at(domn->ngrd-1))*domn->pram->g;
            rhsSrc.at(ipt) /= domn->rho->d.at(ipt);
        }

        if(domn->pram->Lspatial)
            rhsSrc.at(ipt) /= domn->uvel->d.at(ipt);
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! lv mixing term part of the rhs function
 * Calculates the mixing term (Mix) of the right-hand side (Rhs) for the given variable ('uvel', 'vvel' or 'wvel').
 * It considers different boundary conditions, and the characteristics of the domain to compute the mixing terms 
 * for the governing equations.
 * @param gf  \input grid geometric factor for derivatives: (df/dx) = gf * (f - f), i.e. 1/Delta(x) with Delta(x) not constant 
 * @param dxc \input = $\abs{\Delta(x^cCoord)}$, is proportional to cell "volume"
 */

void dv_uvw::getRhsMix(const vector<double> &gf,
                       const vector<double> &dxc){
    /* '&' symbol added to indicate that the input parameters are passed by reference, i.e. the method directly
     * operates on the original object, not a copy of it. This can be more efficient than passing by value 
     * (making a copy), specially when dealing with large objects like vectos, since it avoids the copying overhead.
     * That is to say, if this method modifies the 'gf' and 'dxc' vectors, those modifications will be reflected in
     * the original vectors outside the function (originally defined in the micromixer class)
     * In particular, as this vectors are passed with the 'const' qualifier, the vectors are not modified in this
     * method, but are directly accessed more efficiently without copying them; 'const' is added for safety :)
     */
    // initialize rhsMix to zeros, with size equal to the number of points in the domain
    rhsMix.resize(domn->ngrd, 0.0);

    //------------------ Set fluxes

    // set up variables 'flux' and 'dvisc_f' for future use
    flux.resize(domn->ngrdf);
    vector<double> dvisc_f(domn->ngrdf);

    /* 'interpVarToFacesHarmonic' is an inherited method from 'dv' parent class.
     * It interpolates a cell centered variable to a face
     * by harmonic interpolation which gives roughly an upwind flux
     */
    interpVarToFacesHarmonic(domn->dvisc->d, dvisc_f);

    //---------- Interior faces

    for (int i=1, im=0; i < domn->ngrd; i++, im++)
        flux.at(i) = -gf.at(i) * dvisc_f.at(i)*(d.at(i) - d.at(im));

    //---------- Boundary faces

    if(domn->pram->bcType=="OUTFLOW") {
        flux.at(0) = 0.0;
        flux.at(domn->ngrd) = 0.0;
    }
    // channelFlow: bcType are set to "WALL" in the input.yaml
    else if(domn->pram->bcType=="WALL") {
        /* 'bclo' and 'bcli' are the value of the specific variable (var_name) at the output and input points of the domain,
         * (channelFlow) i.e. at lower and upper boundaries of the y-axis.
         * These values are set using conditional operators
         * e.g. 'bclo':
         * It is calculated based on the values of var_name ('uvel', 'vvel' or 'wvel') and the properties stored
         * in parameters object 'domn->pram'.
         * The conditional operator : ? has the following syntax: condition ? value_if_true : value_if_false
         * Therefore, if    var_name=='uvel'  then 'bclo' is set to domn->pram->uBClo,
         *            elif  var_name=='vvel'  then 'bclo' is set to domn->pram->vBClo,
         *            else (var_name=='wvel') then 'bclo' is set to domn->pram->wBClo,
         */
        double bclo = var_name=="uvel" ? domn->pram->uBClo : var_name=="vvel" ? domn->pram->vBClo : domn->pram->wBClo;
        double bchi = var_name=="uvel" ? domn->pram->uBChi : var_name=="vvel" ? domn->pram->vBChi : domn->pram->wBChi;
        /* The variable flux at the domain boundaries are calculated from the variable boundary values calculated before. 
         * There fluxes account for diffusion, and are calculated by the difference between the variable values at the
         * boundary (actual values odt simulation) and the specified values (bclo, bchi)
         */
        flux.at(0)          = -gf.at(0)          * dvisc_f.at(0)          * (d.at(0) - bclo);
        flux.at(domn->ngrd) = -gf.at(domn->ngrd) * dvisc_f.at(domn->ngrd) * (bchi - d.at(domn->ngrd-1));
    }
    else if(domn->pram->bcType=="WALL_OUT") {
        double bclo = var_name=="uvel" ? domn->pram->uBClo : var_name=="vvel" ? domn->pram->vBClo : domn->pram->wBClo;
        flux.at(0)          = -gf.at(0) * dvisc_f.at(0) * (d.at(0) - bclo);
        flux.at(domn->ngrd) = 0.0;
    }
    else if(domn->pram->bcType=="PERIODIC") {
        int im = domn->ngrd - 1;
        flux.at(0)          = -gf.at(0) * dvisc_f.at(0) * (d.at(0) - d.at(im));
        flux.at(domn->ngrd) = flux.at(0);
    }
    else {
        *domn->io->ostrm << endl << "ERROR: bcType not recognized in dv_uvw::getRhsMix" << endl;
        exit(0);
    }

    //------------------ Compute the mixing term

    // loop over the points of the domain, where 'i' is the index of a point, and 'ip' is the idx of the next point 
    for(int i=0,ip=1; i<domn->ngrd; i++, ip++)
        /* channelFlow: sets cCoord = 1 (planar), therefor the simplified expression of the mixing term is:
         * rhsMix.at(i) = - (1/(domn->rho->d.at(i) * dxc.at(i))) * (flux.at(ip)*1 - flux.at(i)*1)
         */
        rhsMix.at(i) = -domn->pram->cCoord / (domn->rho->d.at(i) * dxc.at(i)) *
                    (flux.at(ip) * pow(abs(domn->posf->d.at(ip)), domn->pram->cCoord-1) -
                     flux.at(i)  * pow(abs(domn->posf->d.at(i) ), domn->pram->cCoord-1));

    // channelFlow: Lspatial = false
    if(domn->pram->Lspatial)
        for(int i=0; i<domn->ngrd; i++)
            rhsMix.at(i) /= domn->uvel->d.at(i);

}

////////////////////////////////////////////////////////////////////////////////
/*! lv statistics convergence term part of the rhs function. 
 *  Method implementation for statistics convergence term of the right-hand side (Rhs) 
 *  @param ipt \input optional point to compute source at.
 */

void dv_uvw::getRhsStatConv(const int ipt) {
    
    if(!L_transported or !L_statconv)
        return;

    rhsStatConv.resize(domn->ngrd, 0.0);
    // TODO: now dvaldt are zeros, implement it!
    dvaldt.resize(domn->ngrd, 0.0);

    // statistics acceleration source term
    if(ipt==-1) {
        for(int i=0; i<domn->ngrd; i++) {
            rhsStatConv.at(i) = dvaldt.at(i);
        }
    }
}


void dv_uvw::updateStatisticsIfNeeded(const double &time, const double &dt) {
    
    double tAvg;
    double dtAvg;
    double timeCurrent;
    vector<double> posCurrent = domn->pos->d;

    
    timeCurrent = time + dt;
    if (timeCurrent > tBeginAvg){ 

        // Interpolate statistics to new grid distribution if needed
        if (~areVectorsEqual(posLast, posCurrent)){
            vector <double> dmb;
            dmb = davg;
            Linear_interp Linterp(posLast, dmb);
            davg.resize(domn->ngrd, 0.0);
            for(int i=0; i<domn->ngrd; i++)
                davg.at(i)= Linterp.interp(posCurrent[i]);
        }

        // calculate averaging time and delta time
        tAvg  = timeCurrent - tBeginAvg;
        dtAvg = tAvg - tLastAvg;

        // update statistic
        for(int k=0; k<domn->v.size(); k++){
            for(int i=0; i<davg.size(); i++) {
                davg.at(i) = ( tLastAvg * davg.at(i) + dtAvg * d.at(i) ) / tAvg;
            }
        } 
        // update time and position quantities
        tLastAvg = tAvg;

    } else {
        davg.resize(domn->ngrd,0.0);
    }
    posLast.resize(domn->ngrd, 0.0);
    posLast = posCurrent;

}


bool dv_uvw::areVectorsEqual(const vector<double> &vec1, const vector<double> &vec2) {
    
    // Check if the sizes are different
    if (vec1.size() != vec2.size()) {
        return false;
    }

    // Compare each element
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec1[i] != vec2[i]) {
            return false; // Elements are different
        }
    }

    // All elements match
    return true;
    
}