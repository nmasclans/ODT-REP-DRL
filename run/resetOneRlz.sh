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

###############################################################################

resetCase () {

    # Format the realizationNum with leading zeros to a total of 5 digits
    formattedRealizationNum=$(printf "%05d" "$realizationNum")
    rm -rf "$ODT_PATH/data/$caseName/data/data_$formattedRealizationNum" > /dev/null 2>&1
    cp     "$inputDir/"*        "$ODT_PATH/data/$caseName/input/" > /dev/null 2>&1

    #--------------------------------------------------------------------------

    echo "*** RESET, NEW REALIZATION ***"
    echo "Output is being written to $ODT_PATH/$caseName/runtime/runtime_* and $ODT_PATH/$caseName/data"
    ./odt.x $caseName $realizationNum          # realizationNum is the shift (realization # here)
    
    #--------------------------------------------------------------------------
    # copy data to 'output' directory for RL framework
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/statistics/stat_odt_end.dat $ODT_PATH/data/$caseName/output/statistics.dat
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/state/state_odt_end.dat $ODT_PATH/data/$caseName/output/state.dat

}

###############################################################################

resetCase

###############################################################################
echo
echo "the end simulation time is"
date
###############################################################################

exit 0