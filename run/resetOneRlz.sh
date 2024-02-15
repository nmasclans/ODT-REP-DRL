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
tEndIncrement=1.0

###############################################################################

resetCase () {

    # Format the realizationNum with leading zeros to a total of 5 digits
    formattedRealizationNum=$(printf "%05d" "$realizationNum")
    rm -rf "../data/$caseName/data/data_$formattedRealizationNum" > /dev/null 2>&1
    cp     "$inputDir/"*        "../data/$caseName/input/" > /dev/null 2>&1

    #--------------------------------------------------------------------------

    echo "*** RESET, NEW REALIZATION ***"
    echo "Output is being written to ../$caseName/runtime/runtime_* and ../$caseName/data"
    ./odt.x $caseName $realizationNum          # realizationNum is the shift (realization # here)

}

###############################################################################

resetCase

###############################################################################
echo
echo "the end simulation time is"
date
###############################################################################

exit 0