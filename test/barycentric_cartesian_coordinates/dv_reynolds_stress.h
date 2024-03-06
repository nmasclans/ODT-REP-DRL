/**
 * @file dv_reynolds_stress.h
 * @brief Header file for class dv_reynolds_stress
 */

#pragma once

#include <string>
#include <vector>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing child dv_reynolds_stress of parent lv object.
 *
 *  @author Nuria Masclans
 */

class dv_reynolds_stress {

    //////////////////// DATA MEMBERS //////////////////////

public:


private:

    vector<double>                  b;
    vector<double>                  t;
    vector<double>                  x1c;
    vector<double>                  x2c;
    vector<double>                  x3c;
    vector<vector<double>>          Binv;
    vector<vector<double>>          Tinv;


    //////////////////// MEMBER FUNCTIONS /////////////////

public:
    
    virtual void getDirectBarycentricMapping(const vector<vector<double>> &Dij, double &xmapping1, double &xmapping2);
    virtual void getInverseBarycentricMapping(const double &xmapping1, const double &xmapping2, vector<vector<double>> &Dij);
    virtual void getBarycentricCoordFromCartesianCoord(const double &xmapping1, const double &xmapping2, vector<double> &lambda);
    virtual void getCartesianCoordFromBarycentricCoord(const vector<double> &lambda, double &xmapping1, double &xmapping2);
    virtual void truncateAndNormalizeBarycentricCoord(vector<double> &lambda);
    virtual void enforceRealizabilityXmap(double &xmap1, double &xmap2);
    virtual bool areElementsInRange(const vector<double> &vec);



    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

public:

    dv_reynolds_stress();
    virtual ~dv_reynolds_stress(){};

};

////////////////////////////////////////////////////////////////////////////////
