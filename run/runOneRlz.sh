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
formattedRealizationNum=$(printf "%05d" "$realizationNum")

###############################################################################

runCase () {

    rm -rf "../data/$caseName" > /dev/null 2>&1
    mkdir  "../data/$caseName"
    mkdir  "../data/$caseName/data"
    mkdir  "../data/$caseName/input"
    mkdir  "../data/$caseName/output"
    mkdir  "../data/$caseName/runtime"
    cp     "$inputDir/"*        "../data/$caseName/input/" > /dev/null 2>&1
    cp -r  "$inputDir/restart"* "../data/$caseName/input/" > /dev/null 2>&1

    #--------------------------------------------------------------------------

    echo "*** RUNNING ***"
    echo "Output is being written to ../$caseName/runtime/runtime_* and ../$caseName/data"
    ./odt.x $caseName $realizationNum          # $realizationNum is the shift (realization # here)

    #--------------------------------------------------------------------------
    # copy data to 'output' directory for RL framework
    cp ../data/$caseName/data/data_$formattedRealizationNum/statistics/stat_odt_end.dat ../data/$caseName/output/statistics.dat
    cp ../data/$caseName/data/data_$formattedRealizationNum/state/state_odt_end.dat ../data/$caseName/output/state.dat

}

###############################################################################

runCase

###############################################################################
echo
echo "the end simulation time is"
date
###############################################################################

exit 0

