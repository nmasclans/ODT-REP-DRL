/**
 * @file processor.cc
 * @brief Source file for class \ref processor
 */
#include "processor.h"
#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////

int processor::nInst;

///////////////////////////////////////////////////////////////////////////////
/** Constructor */

processor::processor() {

#ifdef DOMPI

    //----------- set MPI Stuff (if on)

    if((++nInst)==1) {                 // Only ever call MPI_* once
        int fake_argc = 0;
        char** fake_argv;
        MPI_Init(&fake_argc, &fake_argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    }
    if(nInst > 1)
        cout << endl << "****** WARNING, more than one processor class object" << endl;

    if(myid==0)
        cout << "# MPI IS ON" << "; Nprocs = " << nproc << endl;
    cout << "# MPI myid = " << myid << endl;

    MPI_Barrier(MPI_COMM_WORLD);


#else
    myid  = 0;
    nproc = 1;

#endif

}


///////////////////////////////////////////////////////////////////////////////
/** Destructor */

processor::~processor() {

#ifdef DOMPI

    if((--nInst)==0)             // Only ever finalize mpi once

        // MPI communicator
        MPI_Comm_get_parent(&sub_com);    // define 'sub_com' as the parent communicator (handle) 

        // Synchronization parent (Python) & child (C++)
        MPI_Barrier(sub_com); // C++ barrier that matches python barrier
        cout << "[processor.cc] Barrier passed" << endl;
        
        // Disconnect communication
        if (sub_com != MPI_COMM_NULL)      
            MPI_Comm_disconnect(&sub_com);
        cout << "[processor.cc] MPI Disconnected" << endl;

        // Finalize MPI environment in the current cpu (where C++ is running)
        MPI_Finalize();
        cout << "[processor.cc] MPI Finished" << endl;

#else

    // MPI communicator
    MPI_Comm_get_parent(&sub_com);    // define 'sub_com' as the parent communicator (handle) 

    // Synchronization parent (Python) & child (C++)
    MPI_Barrier(sub_com); // C++ barrier that matches python barrier
    cout << "[processor.cc] Barrier passed" << endl;
    
    // Disconnect communication
    if (sub_com != MPI_COMM_NULL)      
        MPI_Comm_disconnect(&sub_com);
    cout << "[processor.cc] MPI Disconnected" << endl;

    // Finalize MPI environment in the current cpu (where C++ is running)
    MPI_Finalize();
    cout << "[processor.cc] MPI Finished" << endl;

#endif
}

///////////////////////////////////////////////////////////////////////////////





