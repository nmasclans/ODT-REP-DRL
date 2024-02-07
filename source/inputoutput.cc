/*
 * @file inputoutput.cc
 * @brief Source file for class \ref inputoutput
 */

#include "inputoutput.h"
#include "domain.h"
#include "processor.h"
#include <sys/stat.h>               // retrieve information about a file by stat function and structure
#include <iostream>                 // input and output operations: std::cout, std::endl
#include <iomanip>                  // input and output manipulations: std:setfill, std::setw
#include <fstream>
#include <sstream>                  // construct strings with formated data: std::stringstream
#include <cstdlib>
#include <algorithm>                // max_element
#include <dirent.h>                 // directory traversal operations: DIR, dirent, opendir(), readdir(), closedir()

extern processor proc;

///////////////////////////////////////////////////////////////////////////////
/** inputoutput initialization function
 *
 * @param p_domn  \input set domain pointer with.
 */

void inputoutput::init(domain *p_domn) {
    domn    = p_domn;
    LdoDump = false;
}

///////////////////////////////////////////////////////////////////////////////
/** inputoutput constructor function
 *
 * @param p_caseName \input directory with input files
 * @param p_nShift   \input shift the file numbers by this amount (used for multiple sets of parallel runs).
 */

inputoutput::inputoutput(const string p_caseName, const int p_nShift){

    caseName     = p_caseName;
    inputFileDir = "../data/"+caseName+"/input/";

    nShift = p_nShift;

    inputFile   = YAML::LoadFile(inputFileDir+"input.yaml");     ///< use these "nodes" to access parameters as needed
    params      = inputFile["params"];
    streamProps = inputFile["streamProps"];
    initParams  = inputFile["initParams"];
    radParams   = inputFile["radParams"];
    dvParams    = inputFile["dvParams"];
    dTimes      = inputFile["dumpTimes"];
    dumpTimesGen= inputFile["dumpTimesGen"];
    bcCond      = inputFile["bcCond"];

    //--------- setup dumpTimes. Either set the dumpTimesGen parameters or the dumpTimes list directly
    //--------- if dumpTimesGen:dTimeStart is negative, then use the dumpTimes list (if present), otherwise
    //--------- generate the list of dumptimes from the start, stop, and step parameters
    if(dumpTimesGen && dumpTimesGen["dTimeStart"].as<double>() >= 0.0){ // compute dumpTimes
        double DTstart = dumpTimesGen["dTimeStart"].as<double>();
        double DTend   = dumpTimesGen["dTimeEnd"].as<double>();
        double DTstep  = dumpTimesGen["dTimeStep"].as<double>();
        double tEnd    = params["tEnd"].as<double>();
        for(double t=DTstart; t<=tEnd; t+=DTstep){   // note tEnd not DTend
            if(t > DTend)
                break;
            dumpTimes.push_back(t);
        }
    }
    else
        for(int i=0; i<dTimes.size(); i++)
            dumpTimes.push_back(dTimes[i].as<double>());

    dumpTimes.push_back(1.0E200);                       ///< add an extra "infinity" for easy handling of the last time
    iNextDumpTime = 0;

    //----------- set the data directory and runtime file

    string       fname;
    stringstream ss1;
    string       s1;

    ss1.clear(); ss1 << setfill('0') << setw(5) << proc.myid + nShift;
    s1 = ss1.str();
    dataDir     = "../data/"+caseName+"/data/data_" + s1 + "/";   // e.g., "../data_00001", etc.
    dataDirStat = "../data/"+caseName+"/data/data_" + s1 + "/statistics/";

     if (!directoryExists(dataDir) || isDirectoryEmpty(dataDir)) {
        int iflag = mkdir(dataDir.c_str(), 0755);
        if (iflag != 0) {
            cout << "\n********** Error, process " << proc.myid << " failed to create " << dataDir << endl;
            exit(0);
        }
    } else {
        cout << "\nDirectory " << dataDir << " already exists and is not empty. Doing nothing." << endl;
    }

    if (!directoryExists(dataDirStat) || isDirectoryEmpty(dataDirStat)) {
        int iflag = mkdir(dataDirStat.c_str(), 0755);
        if (iflag != 0) {
            cout << "\n********** Error, process " << proc.myid << " failed to create " << dataDirStat << endl;
            exit(0);
        }
    } else {
        cout << "\nDirectory " << dataDirStat << " already exists and is not empty. Doing nothing." << endl;
    }

    fname = "../data/"+caseName+"/runtime/runtime_" + s1;
    ostrm = new ofstream(fname.c_str());

    //----------- set gnuplot file

    // Create & open gnufiles
    
    // -> gnufile: gnuplot script file y vs u, at initial & end time
    fname = dataDir + "plot_odt.gnu";
    gnufile.open(fname.c_str());
    if(!gnufile) {
        cout << endl << "ERROR OPENING FILE: " << dataDir+"plot_odt.gnu" << endl;
        exit(0);
    }

    // -> gnufile_inst: gnuplot script file y vs u,v,w, at each dumpTimes
    fname = dataDir + "plot_instantaneous.gnu";
    gnufile_inst.open(fname.c_str());
    if(!gnufile_inst) {
        cout << endl << "ERROR OPENING FILE: " << dataDir + "plot_instantaneous.gnu" << endl;
        exit(0);
    }
    
    // -> gnufile_stat: gnuplot script file y vs um,vm,wm, at each dumpTimes
    fname = dataDirStat + "plot_statistics.gnu";
    gnufile_stat.open(fname.c_str());
    if(!gnufile_stat) {
        cout << endl << "ERROR OPENING FILE: " << dataDirStat + "plot_statistics.gnu" << endl;
        exit(0);
    }

    // Add axis labels and limits
    gnufile << "set ylabel 'u+'; set xlabel 'y/delta'; set xrange [-1:1]; set yrange [0:25];" << endl;
    gnufile_inst << "set ylabel 'u+'; set xlabel 'y/delta'; set xrange [-1:1]; set yrange [0:25];" << endl; 
    gnufile_stat << "set ylabel 'umean+'; set xlabel 'y/delta'; set xrange [-1:1]; set yrange [0:25];" << endl;

}

///////////////////////////////////////////////////////////////////////////////
/** Destructor function
*/

inputoutput::~inputoutput() {

    delete ostrm;
    gnufile.close();
    gnufile_inst.close();
    gnufile_stat.close();

}

///////////////////////////////////////////////////////////////////////////////
/** Writes a data file of the domain properties (in order)
 *
 * @param fname \input name of the file to write including path
 *  @param time \input time of the output
 */

void inputoutput::outputProperties(const string fname, const double time) {

    string       s1;
    stringstream ss1;

    //--------------------------

    for(int i=0; i<domn->v.size(); i++)
        domn->v.at(i)->setVar();             // make sure the variable is up to date

    //--------------------------

    ofstream ofile(fname.c_str());

    // Filepath and ofstream of statistics from 'fname' filepath of instantaneous data
    size_t dotPos = fname.rfind('.');
    string fnameStat = fname.substr(0,dotPos) + "_stat" +  fname.substr(dotPos);
    // add subdirectory for statistics only
    size_t barPos = fnameStat.rfind("/");
    fnameStat.insert(barPos+1, "statistics/");
    ofstream ofileStat(fnameStat.c_str());

    if(!ofile) {
        *ostrm << "\n\n***************** ERROR OPENING FILE " << fname << endl << endl;
        exit(0);
    }
    if(!ofileStat) {
        *ostrm << "\n\n***************** ERROR OPENING FILE " << fname << endl << endl;
        exit(0);
    }

    *ostrm << endl << "# Writing outputfile: " << fname;
    ostrm->flush();

    //--------------------------

    ofile     << "# time = "   << time;
    ofileStat << "# time = "   << time;

    ofile     << "\n# Grid points = "   << domn->ngrd;
    ofileStat << "\n# FINE UNIFORM Grid points = "   << domn->pram->nunif;

    ofile     << "\n# Pressure (Pa) = " << domn->pram->pres << endl;
    ofileStat << "\n# Pressure (Pa) = " << domn->pram->pres << endl;

    // HEWSON setting tecplot friendly output
    // channelFlow: Ltecplot is set to false
    if (domn->pram->Ltecplot) {
        ofile     << "VARIABLES =";
        ofileStat << "VARIABLES =";
    } else {
        ofile     << "#";
        ofileStat << "#";
    }

    // Write header: text row of variables names, for each output variable column 
    // -> variable names
    int strLength;
    int j = 1;
    for(int i=0; i<domn->v.size(); i++){
        if(domn->v.at(i)->L_output){
            strLength = domn->v.at(i)->var_name.length();
            if (i == 0) {strLength++;}
            if (domn->pram->Ltecplot)
                ofile << setw(18-strLength) << "\"" << j++ << "_" << domn->v.at(i)->var_name << "\"";
            else
                ofile << setw(18-strLength) << j++ << "_" << domn->v.at(i)->var_name;
        }
    }
    // -> statistics names
    j = 1;
    bool isFirstOutputStat = true;
    for(int i=0; i<domn->v.size(); i++){
        if(domn->v.at(i)->L_output_stat){
            if (isFirstOutputStat) {
                string strPosUnif = "posUnif";
                strLength = strPosUnif.length();
                ofileStat << setw(18-(strLength+1)) << j++ << "_" << strPosUnif;
                isFirstOutputStat = false;
            }
            // time-averaged quantity
            string var_name_dmb = domn->v.at(i)->var_name;
            string var_name_avg = var_name_dmb + "_mean"; 
            ofileStat << setw(18-var_name_avg.length()) << j++ << "_" << var_name_avg; 
            // rmsf quantity
            string var_name_rmsf = var_name_dmb + "_rmsf"; 
            ofileStat << setw(18-var_name_rmsf.length()) << j++ << "_" << var_name_rmsf; 
            // F-perturbation for statistics convergence
            string var_name_F_statConv = var_name_dmb + "_Fpert";
            ofileStat << setw(18-var_name_F_statConv.length()) << j++ << "_" << var_name_F_statConv;
        }
    }
    if (domn->Rij->L_output_stat) {
        // Reynolds stress tensor
        ofileStat << setw(15) << j++ << "_Rxx";
        ofileStat << setw(15) << j++ << "_Ryy";
        ofileStat << setw(15) << j++ << "_Rzz";
        ofileStat << setw(15) << j++ << "_Rxy";
        ofileStat << setw(15) << j++ << "_Rxz";
        ofileStat << setw(15) << j++ << "_Ryz";
        // Anisotropy stress - eigenvalues & barycentric map
        ofileStat << setw(11) << j++ << "_lambda0";
        ofileStat << setw(11) << j++ << "_lambda1";
        ofileStat << setw(11) << j++ << "_lambda2";
        ofileStat << setw(13) << j++ << "_xmap1";
        ofileStat << setw(13) << j++ << "_xmap2";
    }

    // Write data
    // -> instantaneous data
    ofile << scientific;
    ofile << setprecision(10);
    for(int i=0; i<domn->ngrd; i++) {
        ofile << endl;
        // -> output data
        for(int k=0; k<domn->v.size(); k++){
            if(domn->v.at(k)->L_output){
                ofile << setw(19) << domn->v.at(k)->d.at(i);
            }
        }
    }
    ofile.close();

    // -> statistics data
    ofileStat << scientific;
    ofileStat << setprecision(10);
    bool isFirstColumn;
    for (int i=0; i<domn->pram->nunif; i++){
        ofileStat << endl;
        isFirstColumn = true;
        for(int k=0; k<domn->v.size(); k++){
            if(domn->v.at(k)->L_output_stat){
                // position
                if (isFirstColumn) {
                    ofileStat << setw(19) << domn->v.at(k)->posUnif.at(i);
                    isFirstColumn = false;
                }
                // time-averaged quantity
                ofileStat << setw(19) << domn->v.at(k)->davg.at(i);
                // rmsf quantity
                ofileStat << setw(19) << domn->v.at(k)->drmsf.at(i);
                // F-perturbation for statistics convergence
                ofileStat << setw(19) << domn->v.at(k)->FstatConvUnif.at(i);
            }
        }
        if (domn->Rij->L_output_stat) {
            // Reynolds stress tensor
            ofileStat << setw(19) << domn->Rij->Rxx.at(i);
            ofileStat << setw(19) << domn->Rij->Ryy.at(i);
            ofileStat << setw(19) << domn->Rij->Rzz.at(i);
            ofileStat << setw(19) << domn->Rij->Rxy.at(i);
            ofileStat << setw(19) << domn->Rij->Rxz.at(i);
            ofileStat << setw(19) << domn->Rij->Ryz.at(i);
            // Anisotropy stress - eigenvalues & barycentric map
            ofileStat << setw(19) << domn->Rij->eigVal.at(i).at(0);
            ofileStat << setw(19) << domn->Rij->eigVal.at(i).at(1);
            ofileStat << setw(19) << domn->Rij->eigVal.at(i).at(2);
            ofileStat << setw(19) << domn->Rij->xmap.at(i).at(0);
            ofileStat << setw(19) << domn->Rij->xmap.at(i).at(1);
        }
    }
    ofileStat.close();

}

///////////////////////////////////////////////////////////////////////////////
/** Set iNextDumpTime from time. Used for restarts.
 * @param time \input time to use to set iNextDumpTime.
 */

void inputoutput::set_iNextDumpTime(double time) {

    for(int i=0; i<dumpTimes.size(); i++) {
        if(dumpTimes[i] > time) {   //set this greater-than dump at the start time
            iNextDumpTime = i;
            break;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////
/** Dumps a domain, sets flag, increments next dump
*/

void inputoutput::dumpDomainIfNeeded(){

    if(!LdoDump) return;

    stringstream ss;
    ss << setfill('0') << setw(5) << iNextDumpTime;
    string fnameRaw = "dmp_" + ss.str() + ".dat";
    string fname    = dataDir + fnameRaw;

    outputProperties(fname, dumpTimes.at(iNextDumpTime));

    iNextDumpTime++;
    LdoDump = false;

    // update gnufile of instantaneous vel. with dump file
    gnufile_inst << "plot '" << fnameRaw << "' us 1:3; pause -1;" << endl;
    // update gnufile of statistics with dump file
    size_t dotPos = fnameRaw.rfind('.');
    string fnameRawStat = fnameRaw.substr(0,dotPos) + "_stat" +  fnameRaw.substr(dotPos);
    gnufile_stat << "plot '" << fnameRawStat << "' us 1:2; pause -1;" << endl;

}

///////////////////////////////////////////////////////////////////////////////
/** Dumps a domain, sets flag, increments next dump
 *  @param fnameRaw \input file name without the path (just the name).
 *  @param time \input time of the output
 */

void inputoutput::writeDataFile(const string fnameRaw, const double time) {

    string fname = dataDir+fnameRaw;
    outputProperties(fname, time);
    gnufile << "plot '" << fnameRaw << "' us 1:3; pause -1;" << endl;

}

///////////////////////////////////////////////////////////////////////////////
/**Output title of properties displayed to screen. */

void inputoutput::outputHeader() {

    *ostrm << endl << "#--------------------------------------------------"
        << "--------------------------------------------------------------------";
    *ostrm << endl;
    *ostrm << setw(5) << "# EE,"
        << setw(12) << "time,"
        << setw(12) << "t-t0,"
        << setw(10) << "nEtry,"
        << setw(6)  << "ngrd,"
        << setw(12) << "edSize,"
        << setw(12) << "edPos,"
        << setw(12) << "edPa,"
        << setw(12) << "nEposs,"
        << setw(12) << "PaAvg,"
        << setw(12) << "invTauEddy"
        ;

    *ostrm << endl << "#--------------------------------------------------"
        << "--------------------------------------------------------------------";
}

///////////////////////////////////////////////////////////////////////////////
/**Outputs the data corresponding to outputHeader.
 * After a given number of accepted eddies, output this info.
 *
 */

void inputoutput::outputProgress() {

    double dmb = 0.5*(domn->ed->leftEdge + domn->ed->rightEdge);
    if(dmb > domn->posf->d.at(domn->ngrd))
        dmb = dmb-domn->Ldomain();

    *ostrm << scientific << setprecision(3) << endl;
    *ostrm << setw(5)  << domn->solv->neddies                 //  1: EE
        << setw(12) << domn->solv->time                       //  2: time
        << setw(12) << domn->solv->time-domn->solv->t0        //  3: t-t0
        << setw(10) << domn->solv->iEtrials                   //  4: nEtry
        << setw(6)  << domn->ngrd                             //  5: ngrd
        << setw(12) << domn->ed->eddySize                     //  6: edSize
        << setw(12) << dmb                                    //  7: edPos
        << setw(12) << domn->ed->Pa                           //  8: edPa
        << setw(12) << domn->solv->nPaSumC                    //  9: nEposs
        << setw(12) << domn->solv->PaSumC/domn->solv->nPaSumC // 10: PaAvg
        << setw(12) << domn->ed->invTauEddy                   // 11: invTauEddy
        ;
    ostrm->flush();
}

///////////////////////////////////////////////////////////////////////////////
/** Restart
 *  The number of columns in the restart file should match the number and order of domainvariables
 *      that are output to a data file. (That is, the order of the domainvariables in domain).
 */

void inputoutput::loadVarsFromRestartFile() {

    string fname, fnameStat;
    stringstream ss1, ss2;
    string       s1,  s2;

    // --- initial checks
    // to restart, a transported variable (L_transported=true) requires saved value (L_output=true) 
    for(int k=0; k<domn->v.size(); k++) {
        if(domn->v[k]->L_transported && !domn->v[k]->L_output) {
            cout << endl << "ERROR: to restart, all transported variables need to be in the restart file" << endl;
            exit(0);
        }
    }

    // --- set restart filepath
    // channelFlow: rstType == "single" by default
    // inputFleDir = "../data/"+caseName+"/input/";
    if(domn->pram->rstType == "multiple") {
        ss1.clear(); ss1 << setfill('0') << setw(5) << proc.myid;
        fname     = inputFileDir + "restart/restart_" + ss1.str() + ".dat";
        fnameStat = inputFileDir + "restart/restartStat_" + ss1.str() + ".dat";
    }
    else // channelFlow
        fname     = inputFileDir + "restart.dat";
        fnameStat = inputFileDir + "restartStat.dat";
    ifstream ifile(fname.c_str());
    ifstream ifileStat(fnameStat.c_str());

    // --- check restart file exists
    if(!ifile) {
        cout << endl << "ERROR: reading restart file " << fname << endl;
        exit(0);
    }
    if(!ifileStat) {
        cout << endl << "ERROR: reading statistics restart file " << fnameStat << endl;
        exit(0);
    }

    //------------- Get file header information

    // --- restart file of instantaneous information
    // set restart time: trst
    getline(ifile, s1);                        // read 1st line "# time = 1.1" (1.1 is the restart time), and stores line in string 's1'
    ss1.clear();                               // clear state flags of the stringstream 'ss1'
    ss1.str(s1);                               // initialize stringstream 'ss1' with the content of string 's1'
    ss1 >> s1 >> s1 >> s1 >> domn->pram->trst; // set domn->pram->trst value as read in restart file
                                               // -> reads four space-separated values from the stringstream 'ss1', and assigns them to string 's1',
                                               // 3 times (essentially skipping the first 3 values), and assigns the 4th value to 'trst' parameter

    // set number of grid points: ngrd & ngrdf
    getline(ifile, s1);                        // read 2nd line "# Grid points = 100"
    ss1.clear();
    ss1.str(s1);
    ss1 >> s1 >> s1 >> s1 >> s1 >> domn->ngrd; // set domn->ngrd, as read in restart file
    domn->ngrdf = domn->ngrd+1;                // set domn->ngrdf, as read in restart file

    // get next lines, information not used
    getline(ifile, s1);                        // read 3rd line "# Pressure (Pa) = 101325
    getline(ifile, s1);                        // read 4th line "# column headers

    // --- restart file of statistical information
    getline(ifileStat, s2);                    // read 1st line (not of use)
    getline(ifileStat, s2);                    // read 2nd line
    getline(ifileStat, s2);                    // read 3rd line
    getline(ifileStat, s2);                    // read 4th line
    

    //------------- Get file data columns

    // --- resize
    // resize domain variables as ngrd, ngrdf
    for(int k=0; k<domn->v.size(); k++)
        domn->v[k]->d.resize(domn->ngrd);
    domn->posf->d.resize(domn->ngrdf);
    // resize statistics variables -> not necessary, contant grid of size domn->pram->nunif

    // --- load values
    // load restart values of domain variables that have L_output = true
    // channelFlow:     domn->v[k]->name:          pos,   posf,  rho,   dvisc, uvel, vvel, wvel
    //                  domn->v[k]->Loutput:       true,  true,  false, false, true, true, true
    //                  domn->v[k]->L_transported: false, false, false, false, true, true, true   
    for(int i=0; i<domn->ngrd; i++) {           // loop over restart file lines, i.e. grid points along domain
        for(int k=0; k<domn->v.size(); k++) {   // loop over domain variables, 
            if(!domn->v[k]->L_output)           // only domain variables with L_output=true, whose values are saved as columns in the restart file
                continue;
            ifile >> domn->v[k]->d[i];          // set domain variables values, for k-th domain varriable, i-th grid point
        }
    }
    domn->posf->d[domn->ngrd] = domn->posf->d[0] + domn->pram->domainLength; // set last domain boundary point of ngrdf, value not saved in restart file 
    
    // load restart values of statistical quantities
    // channelFlow stat columns: 1_posUnif        2_uvel_mean        3_uvel_rmsf       4_uvel_Fpert        5_vvel_mean        6_vvel_rmsf       7_vvel_Fpert        8_wvel_mean        9_wvel_rmsf      10_wvel_Fpert             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz         17_lambda0         18_lambda1         19_lambda2           20_xmap1           21_xmap2
    double aux;
    for (int i=0; i<domn->pram->nunif; i++){
        // pass value 1_posUnif - not used, set properly in dv initialization from pram->nunif
        ifileStat >> aux;
        for (int k=0; k<domn->v.size(); k++){
            if (!domn->v.at(k)->L_output_stat)
                continue;
            ifileStat >> domn->v.at(k)->davg.at(i);
            ifileStat >> domn->v.at(k)->drmsf.at(i);
            ifileStat >> domn->v.at(k)->FstatConvUnif.at(i);
        }
        if (domn->Rij->L_output_stat) {
            // Reynolds stress tensor
            ifileStat >> domn->Rij->Rxx.at(i);
            ifileStat >> domn->Rij->Ryy.at(i);
            ifileStat >> domn->Rij->Rzz.at(i);
            ifileStat >> domn->Rij->Rxy.at(i);
            ifileStat >> domn->Rij->Rxz.at(i);
            ifileStat >> domn->Rij->Ryz.at(i);
            // Anisotropy stress - eigenvalues & barycentric map
            ifileStat >> domn->Rij->eigVal.at(i).at(0);
            ifileStat >> domn->Rij->eigVal.at(i).at(1);
            ifileStat >> domn->Rij->eigVal.at(i).at(2);
            ifileStat >> domn->Rij->xmap.at(i).at(0);
            ifileStat >> domn->Rij->xmap.at(i).at(1);
        }
    }

    //------------- Set the variables

    for(int k=0; k<domn->v.size(); k++)
        domn->v[k]->setVar();                   // channelFlow: only defined for dv_pos (remains idem.), rho_const, dvisc_const (set ct. value as in input file)

}


///////////////////////////////////////////////////////////////////////////////
/* Managing directories
*/

bool inputoutput::directoryExists(const string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool inputoutput::isDirectoryEmpty(const string& path) {
    DIR* dir = opendir(path.c_str());
    if (dir != nullptr) {
        dirent* entry = readdir(dir);
        dirent* nextEntry = readdir(dir);
        closedir(dir);
        return entry == nullptr && nextEntry == nullptr;
    }
    return true; // Assume it's empty if we can't open the directory
}