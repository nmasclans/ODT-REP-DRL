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


private:

    double                          factEigValPert;
    vector<double>                  b;
    vector<double>                  Rkk;
    vector<double>                  x1c;
    vector<double>                  x2c;
    vector<double>                  x3c;
    vector<double>                  xmapTarget;
    vector<double>                  eigValTarget;
    vector<double>                  RxxDeltaUnif;                 
    vector<double>                  RxyDeltaUnif;                 
    vector<double>                  RxzDeltaUnif;                 
    vector<double>                  RyyDeltaUnif;                 
    vector<double>                  RyzDeltaUnif;                 
    vector<double>                  RzzDeltaUnif; 
    vector<vector<double>>          Binv;
    vector<vector<double>>          Deltaij;
    vector<vector<vector<double>>>  eigVect;                


    //////////////////// MEMBER FUNCTIONS /////////////////

public:
    
    virtual void updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time);
    virtual void getReynoldsStressDelta();

private:

    virtual void getDirectBarycentricMapping(const vector<double> &eigenvalues, vector<double> &xmapping);
    virtual void getInverseBarycentricMapping(const vector<double> &xmapping, vector<double> &eigenvalues);
    virtual void getPerturbedTrace(const double &Rkk, double &RkkPert);
    virtual void getPerturbedEigenValuesMatrix(const vector<double> &eigVal, vector<vector<double>> &DijPert);
    virtual void getPerturbedEigenVectorsMatrix(const vector<vector<double>> &eigVect, vector<vector<double>> &QijPert);
    virtual void getPerturbedReynoldsStresses(const double &RkkPert, const vector<vector<double>> &DijPert, const vector<vector<double>> &QijPert, vector<vector<double>> &RijPert);
    virtual void getReynoldsStressesDeltaUnif(const vector<vector<double>> &RijPert, const int &i);
    virtual void interpRijDeltaUniformToAdaptativeGrid();

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
