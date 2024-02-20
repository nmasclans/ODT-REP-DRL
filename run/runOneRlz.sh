#!/bin/bash

# Single realization run
# run as "./runOneRlz.sh" <caseName>
# note it requires a single input argument: caseName

###############################################################################
echo "the start time is"
date
###############################################################################

inputDir="$ODT_PATH/input/channelFlow/$1"
caseName="$1"
realizationNum="$2"
formattedRealizationNum=$(printf "%05d" "$realizationNum")

###############################################################################

runCase () {

    rm -rf "$ODT_PATH/data/$caseName" > /dev/null 2>&1
    mkdir  "$ODT_PATH/data/$caseName"
    mkdir  "$ODT_PATH/data/$caseName/data"
    mkdir  "$ODT_PATH/data/$caseName/input"
    mkdir  "$ODT_PATH/data/$caseName/output"
    mkdir  "$ODT_PATH/data/$caseName/runtime"
    cp     "$inputDir/"*        "$ODT_PATH/data/$caseName/input/" > /dev/null 2>&1
    cp -r  "$inputDir/restart"* "$ODT_PATH/data/$caseName/input/" > /dev/null 2>&1

    #--------------------------------------------------------------------------

    echo "*** RUNNING ***"
    echo "Output is being written to $ODT_PATH/$caseName/runtime/runtime_* and $ODT_PATH/$caseName/data"
    $ODT_PATH/run/odt.x $caseName $realizationNum          # $realizationNum is the shift (realization # here)

    #--------------------------------------------------------------------------
    # copy data to 'output' directory for RL framework
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/statistics/stat_odt_end.dat $ODT_PATH/data/$caseName/output/statistics.dat
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/state/state_odt_end.dat $ODT_PATH/data/$caseName/output/state.dat

}

###############################################################################

runCase

###############################################################################
echo
echo "the end simulation time is"
date
###############################################################################

exit 0

