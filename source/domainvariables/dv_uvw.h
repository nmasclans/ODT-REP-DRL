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

        double                        tLastAvg;       ///< time of last statistics update
        double                        tBeginAvg;      ///< averaging time at which to begin calculating statistics
        vector<double>                posLast;        ///< todo: add description
    
    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        virtual void getRhsSrc(const int ipt=-1);
        virtual void getRhsMix(const vector<double> &gf,
                               const vector<double> &dxc);
        virtual void getRhsStatConv(const int ipt=-1);
        virtual void updateStatisticsIfNeeded(const double &time, const double &dt);

    private:

        bool areVectorsEqual(const vector<double> &vec1, const vector<double> &vec2);

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

