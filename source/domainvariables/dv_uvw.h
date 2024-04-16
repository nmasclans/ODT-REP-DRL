/**
 * @file dv_uvw.h
 * @brief Header file for class dv_uvw
 */

#pragma once

#include "dv.h"
#include <string>
#include <vector>
#include <ostream>
#include <fstream>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing child dv_uvw of parent lv object.
 *
 *  @author David O. Lignell
 */

class dv_uvw : public dv {

    //////////////////// DATA MEMBERS //////////////////////


    //////////////////// MEMBER FUNCTIONS /////////////////

    public:

        virtual void getRhsSrc(const int ipt=-1);
        virtual void getRhsMix(const vector<double> &gf, const vector<double> &dxc);
        virtual void getRhsStatConv(const vector<double> &gf, const vector<double> &dxc, const double &time);
        virtual void updateTimeAveragedQuantities(const double &delta_t, const double &averaging_time, const double &time);
        
    private:

        double controller_output;
        double controller_error;
        double controller_K_p;
        double halfChannel;
        string odtPath;
        string fname;
        ofstream *ostrm;
        double u_bulk;

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        void getOdtPath(string &odtPath);

        dv_uvw(){}      
        dv_uvw(domain      *line,
               const string s,
               const bool   Lt,
               const bool   Lo=true,
               const bool   Lcs=false);

        virtual ~dv_uvw(){}

};


////////////////////////////////////////////////////////////////////////////////

