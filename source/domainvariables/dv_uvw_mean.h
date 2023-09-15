/**
 * @file dv_uvw_mean.h
 * @brief Header file for class dv_uvw_mean
 */

#pragma once

#include "dv.h"
#include <string>
#include <vector>

class domain;

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/** Class implementing child dv_temp of parent lv object.
 *
 *  @author Núria Masclans
 */

class dv_uvw_mean : public dv {


    //////////////////// DATA MEMBERS //////////////////////

    public: 
        string      var_name_inst;  ///< var_name of corresponding instantaneous velocity component

    private:
        double      tLast;          ///< time of last statistics update

    //////////////////// MEMBER FUNCTIONS /////////////////

    public:
        virtual void setVar(const int ipt=-1);
        virtual void update(const double &time);

    private:

    //////////////////// CONSTRUCTOR FUNCTIONS /////////////////

    public:

        dv_uvw_mean(){}
        dv_uvw_mean(domain      *line,
                    const string s,
                    const bool   Lt,
                    const bool   Lo=true);

        virtual ~dv_uvw_mean(){}

};


////////////////////////////////////////////////////////////////////////////////

