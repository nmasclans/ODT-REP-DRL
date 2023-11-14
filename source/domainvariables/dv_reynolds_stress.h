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

/** Class implementing child dv_reynolds_stress of parent lv object.
 *
 *  @author Nuria Masclans
 */

class dv_reynolds_stress : public dv
{

    //////////////////// DATA MEMBERS //////////////////////

public:

    vector<vector<vector<double>>> RijDelta;

private:

    double                  factorEigValPert;
    vector<double>          b;
    vector<double>          Rkk;
    vector<double>          x1c;
    vector<double>          x2c;
    vector<double>          x3c;
    vector<double>          xmap_target;
    vector<vector<double>>  Binv;
    vector<vector<double>>  Deltaij;
    vector<vector<double>>  eigVect;

    //////////////////// MEMBER FUNCTIONS /////////////////

public:
    
    virtual void updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time);
    virtual void getReynoldsStressDelta(const double &k_factor, const double &eigvect_rot, const double &eigval_factor);

private:
    
    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

public:

    dv_reynolds_stress(){};
    dv_reynolds_stress(domain *line,
                       const string s,
                       const bool Lt,
                       const bool Lo = true);

    virtual ~dv_reynolds_stress(){};

};

////////////////////////////////////////////////////////////////////////////////
