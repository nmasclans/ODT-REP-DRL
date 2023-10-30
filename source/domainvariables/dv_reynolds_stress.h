/**
 * @file dv_reynolds_stress.h
 * @brief Header file for class dv_reynolds_stress
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

class dv_reynolds_stress : public dv {

    //////////////////// DATA MEMBERS //////////////////////

    public:

        vector<double>                Rxx;                    ///< todo: add description
        vector<double>                Rxy;                    ///< todo: add description
        vector<double>                Rxz;                    ///< todo: add description
        vector<double>                Ryy;                    ///< todo: add description
        vector<double>                Ryz;                    ///< todo: add description
        vector<double>                Rzz;                    ///< todo: add description

    private: 


    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        virtual void updateReynoldsStress(const double &delta_t, const double &averaging_time);
        
    private:

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        dv_reynolds_stress(){};
        dv_reynolds_stress(domain      *line,
                           const string s,
                           const bool   Lt,
                           const bool   Lo=true,
                           const bool   Lcs=false);

        virtual ~dv_reynolds_stress(){};

};


////////////////////////////////////////////////////////////////////////////////

