/**
 * @file main.cc
 * @brief main driver for ODT 
 */

#include "domain.h"
#include "streams.h"
#include "processor.h"
#include "param.h"
#include "micromixer.h"
#include "meshManager.h"
#include "eddy.h"
#include "solver.h"
#include "randomGenerator.h"
#include "cantera/thermo/IdealGasPhase.h"
#include "cantera/transport.h"

#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

// namespaces 'std' and 'Cantera' are used for ease of access to standard and Cantera library functions and classes
using namespace std;
using namespace Cantera;

//////////////////////////////////////////////////////////////

// global processor object: an instance of the 'processor' class named 'proc' is declared globally
processor proc;

//////////////////////////////////////////////////////////////

/*! Main function:
 * takes command-line arguments 'argc' (argument count) and 'argv' (argument vectors),
 * defined when executing the program as:
 * ./odt.x argument1 argument2, where automatically included argument0 tends to be the exacutable filename 
 */
int main(int argc, char*argv[]) {

    // Command-line arguments check: implementation requires 3 arguments 
    if(argc<3) {
        cout << endl << "ERROR: code needs caseName and shift arguments" << endl; // character output by std:cout, from <iostream> library, part of 'std' namespace
        return 1;                                                                 // indicating an error
    }
    
    // Arguments extraction
    string caseName= argv[1];           // example: temporalJet (../input/temporalJet, without the ../input/)

    int nShiftFileNumbers = 0;
    stringstream ss1;                   // 'stringstream' object is a stream that operates on strings, used to convert between strings and other data types     
    ss1.clear(); ss1 << argv[2];
    ss1 >> nShiftFileNumbers;

    // Objects creation
    inputoutput           io(caseName, nShiftFileNumbers);
    param                 pram(&io);
    streams               strm;
    IdealGasPhase         gas("../input/gas_mechanisms/"+pram.chemMechFile);
    Transport             *tran = newTransportMgr("Mix", &gas);
    eddy                  ed;
    meshManager           mesher;
    solver                *solv;
    micromixer            *mimx;
    eigenDecomposition    eigdec;
    solv = new solver();
    mimx = new micromixer();

    domain domn(NULL,  &pram);
    domain eddl(&domn, &pram);

    // we should increment the seed if we are starting MPI multiple times
    if ( pram.seed >= 0 ) pram.seed += nShiftFileNumbers;
    randomGenerator rand(pram.seed);

    // Domain and eddy domain initialization
    domn.init(&io,  &mesher, &strm, &gas, tran, mimx, &ed, &eddl, solv, &rand, &eigdec);
    eddl.init(NULL, NULL,    NULL,  NULL, NULL, NULL, NULL,NULL,  NULL, NULL,  NULL,   true);
    
    //-------------------

    // Starting time
    time_t mytimeStart, mytimeEnd;
    mytimeStart = time(0);
    *io.ostrm << endl << "#################################################################";
    *io.ostrm << endl << "#  Start Time = " << ctime(&mytimeStart);
    *io.ostrm << endl << "#################################################################";


    //-------------------

    // Solution calculation
    domn.solv->calculateSolution();

    //domn.io->outputProperties("../data/init.dat", 0.0); //doldb
    //domn.mimx->advanceOdt(0.0, domn.pram->tEnd);        //doldb
    //delete mimx;
    //delete solv;

    //-------------------

    // Ending time
    mytimeEnd = time(0);
    *io.ostrm << endl << "#################################################################";
    *io.ostrm << endl << "#  Start Time = " << ctime(&mytimeStart);
    *io.ostrm << endl << "#  End Time   = " << ctime(&mytimeEnd);
    *io.ostrm << endl << "#################################################################";
    *io.ostrm << endl;

    return 0;           // 0 indicates successful execution 


}
