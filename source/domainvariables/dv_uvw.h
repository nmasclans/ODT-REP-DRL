/**
 * @file dv_uvw.h
 * @brief Header file for class dv_uvw
 */

#pragma once

#include "dv.h"
#include <string>
#include <vector>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing child dv_uvw of parent lv object.
 *
 *  @author David O. Lignell
 */

class dv_uvw : public dv {

    //////////////////// DATA MEMBERS //////////////////////

    public:

    private: 


    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        virtual void getRhsSrc(const int ipt=-1);
        virtual void getRhsMix(const vector<double> &gf,
                               const vector<double> &dxc);
        virtual void getRhsStatConv(const double &timeCurrent);
        virtual void updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time);
        virtual double updateTimeMeanQuantity(const double &quantity, const double &mean_quantity, const double &delta_t, const double &averaging_time);
        virtual double updateTimeRmsfQuantity(const double &quantity, const double &mean_quantity, const double &rmsf_quantity, const double &delta_t, const double &averaging_time);
        
    private:

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        dv_uvw(){}      
        dv_uvw(domain      *line,
               const string s,
               const bool   Lt,
               const bool   Lo=true,
               const bool   Lcs=false);

        virtual ~dv_uvw(){}

};


////////////////////////////////////////////////////////////////////////////////

