#!/bin/bash

# Single realization run
# run as "./runOneRlz.sh" <caseName>
# note it requires a single input argument: caseName

###############################################################################
echo "the start time is"
date
###############################################################################

inputDir="../input/channelFlow/$1"
caseName="$1"
realizationNum="$2"

###############################################################################

runCase () {

    rm -rf "../data/$caseName" > /dev/null 2>&1
    mkdir  "../data/$caseName"
    mkdir  "../data/$caseName/data"
    mkdir  "../data/$caseName/input"
    mkdir  "../data/$caseName/runtime"
    cp     "$inputDir/"*        "../data/$caseName/input/" > /dev/null 2>&1
    cp -r  "$inputDir/restart"* "../data/$caseName/input/" > /dev/null 2>&1

    #--------------------------------------------------------------------------

    echo "*** RUNNING ***"
    echo "Output is being written to ../$caseName/runtime/runtime_* and ../$caseName/data"
    ./odt.x $caseName $realizationNum          # $realizationNum is the shift (realization # here)

}

###############################################################################

runCase

###############################################################################
echo
echo "the end simulation time is"
date
###############################################################################

exit 0

