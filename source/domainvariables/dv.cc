/**
 * @file dv.cc
 * @brief Source file for class \ref dv
 */

#include "dv.h"
#include "domain.h"
#include "interp_linear.h"
#include <iostream>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! dv constructor function
 *
 * @param p_domn  \input set domain pointer with.
 * @param p_phi   \input set vector pointer with.
 */

dv::dv(domain    *line,
       const      string s,
       const bool Lt,
       const bool Lo) {

    domn          = line;
    var_name      = s;
    L_transported = Lt;
    L_output      = Lo;
    d             = vector<double>(domn->ngrd, 0.0);
    
    tfRL          = domn->pram->trst + domn->pram->dtActionRL;

    LagSrc = false;

    // additional setup - only used in dv_uvw
    L_output_stat   = false;
    L_converge_stat = false;

    // position uniform fine grid
    nunif           = domn->pram->nunif;
    posUnif         = vector<double>(nunif, 0.0);     // uniform grid in y-axis
    double delta    = domn->pram->domainLength / 2;   // half-channel length 
    for (int i=0; i<nunif; i++) {
        posUnif.at(i) = - delta + i * (2.0 * delta) / (nunif - 1);
    }

}

////////////////////////////////////////////////////////////////////////////////
/*! dv splitCell function
 *
 * @param isplt  \input index of cell to split
 * @param nsplt  \input number of cells to split cell into
 * @param cellFaces \input original left edge, new interior faces, orig. right edge.
 */

void dv::splitCell(const int isplt,
                   const int nsplt,
                   const vector<double> &cellFaces) {

    d.insert( d.begin() + isplt, nsplt, d.at(isplt) );

}

////////////////////////////////////////////////////////////////////////////////
/*! dv merger2cells function
 *
 * Function presumes that the variable being merged is a quantity per unit mass.
 * Merging conservatively: (rho*phi*dx)_merged = (rho*phi*dx)_1 + (rho*phi*dx)_2
 * Then solve for phi_merged.
 *
 * @param imrg \input merge cells imrg and imrg+1
 * @param m1   \input mass in cell imrg
 * @param m2   \input mass in cell imrg
 * @param LconstVolume \input (for posf, default is false)
 */

void dv::merge2cells(const int    imrg,
                     const double m1,
                     const double m2,
                     const bool   LconstVolume) {

    d.at(imrg) = (d.at(imrg)*m1 + d.at(imrg+1)*m2 ) / (m1+m2);

    d.erase(d.begin() + imrg+1);

}

////////////////////////////////////////////////////////////////////////////////
/*! interpolate a cell centered variable to a face
 *  by harmonic interpolation which gives roughly an upwind flux
 */
void dv::interpVarToFacesHarmonic(const vector<double> &cvar, vector<double> &fvar){

    // todo: fill this in

    double dx1, dx2, k1, k2;
    int i, im;

    double dfirst;      // store the first face diff till end
    double dlast;       // store the last face diff till end
    double denom;

    //------ do edges

    if (domn->pram->Lperiodic) {
        i = 0;
        im = domn->ngrd - 1;

        dx1 = domn->posf->d.at(domn->ngrd) - domn->pos->d.at(im);
        dx2 = domn->pos->d.at(i) - domn->posf->d.at(i);
        k1 = cvar.at(im);
        k2 = cvar.at(i);
        denom = k1*dx2+k2*dx1;
        if(abs(denom)==0.0)
            dfirst = 0.0;
        else
            dfirst = k1 * k2 * (dx1 + dx2) / denom; // first face
        dlast = dfirst; // last face
    }
    else {
        dfirst = cvar.at(0); // first face
        dlast = cvar.at(domn->ngrd-1); // last face
    }

    //------ do interior faces
    // we assume pos[i] is located right in the middle of posf[i] and posf[i+1]
    // so we take always the left half of cells

    for (i=1, im=0; i < domn->ngrd; i++, im++) {
        dx1 = domn->pos->d.at(im) - domn->posf->d.at(im);
        dx2 = domn->pos->d.at(i) - domn->posf->d.at(i);
        k1 = cvar.at(im);
        k2 = cvar.at(i);
        denom = k1*dx2+k2*dx1;
        if(abs(denom)==0.0)
            fvar.at(i) = 0.0;
        else
            fvar.at(i) = k1 * k2 * (dx1 + dx2) / denom;
    }

    fvar.at(0)          = dfirst; // insert the first face flux
    fvar.at(domn->ngrd) = dlast;  // insert the last face flux

}

///////////////////////////////////////////////////////////////////////////////
/** Interpolate variables to single face
 *
 * @param iface \input face index to interpolate to.
 * @param vec   \input variable being interpolated.
 * @return return interpolated variable at desired face.
 */
double dv::linearInterpToFace(const int &iface, const vector<double> &vec) {

    double x1, x2, y1, y2;

    if (iface == 0) {
        if (domn->pram->Lperiodic) {
            x1 = domn->pos->d.at(domn->ngrd - 1) - domn->Ldomain();
            x2 = domn->pos->d.at(0);
            y1 = vec.at(domn->ngrd - 1);
            y2 = vec.at(0);
        } else {
            return vec.at(0);
        }
    } else if (iface == domn->ngrd) {
        if (domn->pram->Lperiodic) {
            x1 = domn->pos->d.at(domn->ngrd - 1);
            x2 = domn->pos->d.at(0) + domn->Ldomain();
            y1 = vec.at(domn->ngrd - 1);
            y2 = vec.at(0);
        } else {
            return vec.at(domn->ngrd - 1);
        }
    } else {
        x1 = domn->pos->d.at(iface - 1);
        x2 = domn->pos->d.at(iface);
        y1 = vec.at(iface - 1);
        y2 = vec.at(iface);
    }

    return y1 + (y2 - y1) / (x2 - x1) * (domn->posf->d.at(iface) - x1);
}

/////////////////////////////////////////////////////////////////////////////////
/*! Interpolate variable between different grids
*/

// Interpolate variable from adaptative grid to uniform fine grid
/*
void dv::interpVarAdaptToUnifGrid(const vector<double> &dAdapt, vector<double> &dUnif){
    //// add wall values, where pos(bw) = -1, pos(tw) = 1, dAdapt(bw,tw) = 0
    size_t nAdapt = dAdapt.size();
    // Create dmb with 2 extra
    vector<double> var_dmb(nAdapt + 2, 0.0);
    // Copy dAdapt into dmb starting from index 1
    copy(dAdapt.begin(), dAdapt.end(), var_dmb.begin() + 1); 
    vector<double> pos_dmb(nAdapt + 2, 0.0);
    pos_dmb.at(0)        = - domn->pram->domainLength * 0.5;
    pos_dmb.at(nAdapt-1) = domn->pram->domainLength * 0.5;
    copy(domn->pos->d.begin(), domn->pos->d.end(), pos_dmb.begin() + 1); 
    Linear_interp Linterp(pos_dmb, var_dmb);
    for (int i=0; i<nunif; i++) {
        // velocity instantaneous (fine grid)
        dUnif.at(i) = Linterp.interp(posUnif.at(i));
    } 
}
*/

void dv::interpVarAdaptToUnifGrid(const vector<double> &dAdapt, vector<double> &dUnif){
    vector<double> dmb = dAdapt;
    Linear_interp Linterp(domn->pos->d, dmb);
    for (int i=0; i<nunif; i++) {
        // velocity instantaneous (fine grid)
        dUnif.at(i) = Linterp.interp(posUnif.at(i));
    } 
}



// Interpolate variable from uniform fine grid to adaptative grid
void dv::interpVarUnifToAdaptGrid(const vector<double> &dUnif, vector<double> &dAdapt){
    dAdapt.resize(domn->ngrd);
    vector<double> dmb = dUnif;
    Linear_interp Linterp(posUnif, dmb);
    for (int i=0; i<domn->ngrd; i++) {
        // velocity instantaneous (fine grid)
        dAdapt.at(i) = Linterp.interp(domn->pos->d.at(i));
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Set data array from region of domain.
 *  @param i1 \input index of starting cell of domn to build from
 *  @param i2 \input index of ending cell of domn to build from
 *  See domain::setDomainFromRegion for additional details.
 */

void dv::setDvFromRegion(const int i1, const int i2){

    // note, we are owned by the eddyline, so domn is eddl, so to get domn data, use domn->domn
    const vector<double> &domn_data = domn->domn->varMap.find(var_name)->second->d;

    if(i2 >= i1)
        d.assign(domn_data.begin()+i1, domn_data.begin()+i2+1  );
    else {           // wrap around (periodic assignment)
        d.assign(domn_data.begin()+i1, domn_data.end());
        d.insert(d.end(), domn_data.begin(), domn_data.begin()+i2+1 );
    }

}


////////////////////////////////////////////////////////////////////////////////
/*! Resize data
 */

void dv::resize() {
    d.resize(domn->ngrd);
}


////////////////////////////////////////////////////////////////////////////////
/*! lv statistics convergence term part of the rhs function. 
 *  Method implementation for statistics convergence term of the right-hand side (Rhs) 
 *  @param timeCurrent \input current time.
 *  @param ipt \input optional point to compute source at.
 */

void dv::getRhsStatConv(const vector<double> &gf, const vector<double> &dxc, const double &time){
    
    if(!L_transported) return;

    rhsStatConv.resize(domn->ngrd, 0.0);

}

////////////////////////////////////////////////////////////////////////////////
/*! Update time-average and rmsf statistics quantities
 */

double dv::updateTimeMeanQuantity(const double &quantity, const double &mean_quantity, const double &delta_t, const double &averaging_time) {

    double updated_mean_quantity = ( averaging_time * mean_quantity + delta_t * quantity ) / (averaging_time + delta_t);

    return( updated_mean_quantity );

}

double dv::updateTimeRmsfQuantity(const double &quantity, const double &mean_quantity, const double &rmsf_quantity, const double &delta_t, const double &averaging_time) {

    double updated_rmsf_quantity = sqrt( ( pow( rmsf_quantity, 2.0 )*averaging_time + pow( quantity - mean_quantity, 2.0 )*delta_t )/( averaging_time + delta_t ) ); 

    return( updated_rmsf_quantity );

}


////////////////////////////////////////////////////////////////////////////////
/*! Print variables for testing - scalars, vectors and matrices
 */

void dv::coutScalar(const double varValue, const string varName){
    cout << varName << ": " << varValue << endl;
}

void dv::coutVector(const vector<double> varValue, const string varName){
    cout << varName << ": " << endl;
    for (int j=0; j<varValue.size(); j++)
        cout << varValue[j] << ", ";
    cout << endl;
}

void dv::coutMatrix(const vector<vector<double>> varValue, const string varName){
    cout << varName << ": " << endl;
    for (int j=0; j<varValue.size(); j++){
        for (int k=0; k<varValue[j].size(); k++){
            cout << varValue[j][k] << ", ";
        }
        cout << endl;        
    }
}

////////////////////////////////////////////////////////////////////////////////
void dv::setModifiedParams(){
    // for updating tfRL after restart for both domn->dv_uvw and eddy->dv_uvw, according to restart time read in restart files (in inputoutput.cc)
    tfRL = domn->pram->trst + domn->pram->dtActionRL;
}