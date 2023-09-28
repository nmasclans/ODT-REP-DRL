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

        double                   tLastAvg;                    ///< time of last statistics update
        double                   tBeginAvg;                   ///< averaging time at which to begin calculating statistics
        bool                     gridStatisticsEverUpdated;   ///< todo: add description
        int                      nunif;                       ///< todo: add description
        double                   time_statConv;
        double                   time_statConvLast;
        vector<double>           Favg_statConv;
        vector<double>           Favg_statConvLast;

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        virtual void getRhsSrc(const int ipt=-1);
        virtual void getRhsMix(const vector<double> &gf,
                               const vector<double> &dxc);
        virtual void getRhsStatConv(const double &timeCurrent, const int ipt=-1);
        virtual void updateStatistics(const double &timeCurrent);

    private:

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        dv_uvw(){}      
        dv_uvw(domain      *line,
               const string s,
               const bool   Lt,
               const bool   Lo=true,
               const bool   Lsc=false);

        virtual ~dv_uvw(){}

};


////////////////////////////////////////////////////////////////////////////////

