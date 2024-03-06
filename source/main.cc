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
#include <mpi.h>

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

    // MPI arguments
    //// cout << "[main.cc] Application name: " << argv[0] << endl;
    //// cout << "[main.cc] Application arguments: there are " << argc -1 << " input arguments" << endl;
    //// cout << "[main.cc]     Case name: " << argv[1] << endl;
    //// cout << "[main.cc]     Realization number: " << argv[2] << endl;
    
    // MPI environment
    MPI_Init(&argc, &argv);           // initialize MPI execution environment
    
    // MPI communicator
    MPI_Comm sub_com;                 // initialize MPI_Comm object (logical group of MPI processes)
    MPI_Comm_get_parent(&sub_com);    // define 'sub_com' as the parent communicator (handle) 
    
    //-------------------    

    // Command-line arguments check: implementation requires 3 arguments 
    if(argc<3) {
        cout << endl << "ERROR: code needs caseName and shift arguments" << endl; // character output by std:cout, from <iostream> library, part of 'std' namespace
        return 1;                                                                 // indicating an error
    }

    // export ODT_PATH environment variable
    string odtPath; 
    char* odtPath_ = getenv("ODT_PATH");
    if (odtPath_ != nullptr) {
        odtPath = odtPath_;
    } else {
        throw runtime_error("ODT_PATH environment variable is not set.");
    }
    
    // Arguments extraction
    string caseName= argv[1];           // example: temporalJet (<odtPath>/input/temporalJet, without the <odtPath>/input/)

    int nShiftFileNumbers = 0;
    stringstream ss1;                   // 'stringstream' object is a stream that operates on strings, used to convert between strings and other data types     
    ss1.clear(); ss1 << argv[2];
    ss1 >> nShiftFileNumbers;

    // Objects creation
    inputoutput           io(caseName, nShiftFileNumbers);
    param                 pram(&io);
    streams               strm;
    IdealGasPhase         gas(odtPath+"/input/gas_mechanisms/"+pram.chemMechFile);
    Transport             *tran = newTransportMgr("Mix", &gas);       // for Cantera version < 3.0.0, for Cantera 3.0.0 generates DeprecationWarning 
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

    //domn.io->outputProperties "<odtPath>/data/init.dat", 0.0); //doldb
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

    //-------------------

    // Synchronization parent (Python) & child (C++)
    MPI_Barrier(sub_com); // C++ barrier that matches python barrier
    cout << "[main.cc] Barrier passed" << endl;
    
    // Disconnect communication
    if (sub_com != MPI_COMM_NULL)      
        MPI_Comm_disconnect(&sub_com);
    cout << "[main.cc] MPI Disconnected" << endl;

    // Finalize MPI environment in the current cpu (where C++ is running)
    MPI_Finalize();
    cout << "[main.cc] MPI Finished" << endl;

    //-------------------

    return 0;           // 0 indicates successful execution 

}