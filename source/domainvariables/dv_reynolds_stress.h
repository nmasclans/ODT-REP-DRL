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

        vector<double>                  b;
        vector<double>                  t;
        vector<double>                  x1c;
        vector<double>                  x2c;
        vector<double>                  x3c;
        vector<double>                  RxxDeltaUnif;                 
        vector<double>                  RxyDeltaUnif;                 
        vector<double>                  RxzDeltaUnif;                 
        vector<double>                  RyyDeltaUnif;                 
        vector<double>                  RyzDeltaUnif;                 
        vector<double>                  RzzDeltaUnif; 
        vector<vector<double>>          Binv;
        vector<vector<double>>          Tinv;
        vector<vector<double>>          Deltaij;
        vector<vector<vector<double>>>  eigVect;                


        //////////////////// MEMBER FUNCTIONS /////////////////

    public:
        
        virtual void updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time);
        virtual void getReynoldsStressDelta();

    private:

        virtual void getDirectBarycentricMapping(const vector<vector<double>> &Dij, double &xmapping1, double &xmapping2);
        virtual void getInverseBarycentricMapping(const double &xmapping1, const double &xmapping2, vector<vector<double>> &Dij);
        virtual void getEulerAnglesFromRotationMatrix(const vector<vector<double>> &rotationMatrix, double &thetaZ_i, double &thetaY_i, double &thetaX_i);
        virtual void getRotationMatrixFromEulerAngles(const double &thetaZ_i, const double &thetaY_i, const double &thetaX_i, vector<vector<double>> &rotationMatrix);    
        virtual void getPerturbedReynoldsStresses(const double &RkkPert, const vector<vector<double>> &DijPert, const vector<vector<double>> &QijPert, vector<vector<double>> &RijPert);
        virtual void getReynoldsStressesDeltaUnif(const vector<vector<double>> &RijPert, const int &i);
        virtual void interpRijDeltaUniformToAdaptativeGrid();
        
        // enforce realizability of perturbed xmap
        virtual void getBarycentricCoordFromCartesianCoord(const double &xmapping1, const double &xmapping2, vector<double> &lambda);
        virtual void getCartesianCoordFromBarycentricCoord(const vector<double> &lambda, double &xmapping1, double &xmapping2);
        virtual bool areRealizableBarycentricCoord(const vector<double> &lambda);
        virtual void truncateAndNormalizeBarycentricCoord(vector<double> &lambda);
        virtual void enforceRealizabilityRkk(double &rkk);
        virtual void enforceRealizabilityXmap(double &xmap1, double &xmap2);

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
