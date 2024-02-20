#!/bin/bash

# Single realization run
# run as "./runOneRlz.sh" <caseName> <realizationNum> <tEndIncrement>
# note it requires a single input argument: caseName

###############################################################################
echo "the start time is"
date
###############################################################################

inputDir="$ODT_PATH/input/channelFlow/$1"
caseName="$1"
realizationNum="$2"
tEndIncrement="$3"
formattedRealizationNum=$(printf "%05d" "$realizationNum")

###############################################################################

evolveCase () {

    # Task 1: Set Lrestart to true in input.yaml
    input_yaml_path="$ODT_PATH/data/$caseName/input/input.yaml"
    sed -i 's/Lrestart:.*/Lrestart:       true/' "$input_yaml_path"

    # Task 2: Find and copy the end files in the data folder to 'input' folder
    cp  $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/odt_end.dat $ODT_PATH/data/$caseName/input/restart.dat
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/statistics/stat_odt_end.dat $ODT_PATH/data/$caseName/input/restartStat.dat
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/state/state_odt_end.dat $ODT_PATH/data/$caseName/input/restartState.dat
    
    # Task 3: Copy the action values file from 'input' folder to the data folder, for later inverstigation
    dmp_last=$(ls -1v $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/dmp_*.dat | grep -oE '[0-9]+' | tail -n 1) 
    cp $ODT_PATH/data/$caseName/input/restartAction.dat $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/action/action_dmp_$dmp_last.dat 

    # Task 4: Extract and compare the time value of last data snapshot with tEnd input parameter
    # -> tEnd from data
    tEnd_data=$(head -n 1 $ODT_PATH/data/$caseName/input/restart.dat | grep -oP '# time = \K[0-9.]+(\.[0-9]+)?')
    # Check if there is no decimal point, append ".0"
    if ! [[ "$tEnd_data" =~ \. ]]; then
        tEnd_data="$tEnd_data.0"
    fi
    # > tEnd from input
    tEnd_input=$(awk '/tEnd:/ {print $2}' "$input_yaml_path")
    # Check if tEnd is a valid number
    if ! [[ "$tEnd_input" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: Invalid tEnd format in input.yaml"
        exit 1
    fi
    # Check if tEnd from data & input are equal
    compare_result=$(echo "$tEnd_data == $tEnd_input" | bc)
    if [ "$compare_result" -ne 1 ]; then
        echo "Error: Time value in restart file does not match params.tEnd in input.yaml"
        exit 1
    fi
    # Increment tEnd by tEndIncrementç
    tEnd_new=$(echo "$tEnd_input + $tEndIncrement" | bc)
    # Update tEnd value in input.yaml
    sed -i "s/tEnd:           $tEnd_input/tEnd:           $tEnd_new/" "$input_yaml_path"
    
    # Task 5: update dTimeEnd as tEnd, to produce output data for additional simulation time
    dTimeEnd=$(awk '/dTimeEnd:/ {print $2}' "$input_yaml_path")
    sed -i "s/dTimeEnd:       $dTimeEnd/dTimeEnd:       $tEnd_new/" "$input_yaml_path"

    # Task 6: remove output from previous ODT simulation
    rm "$ODT_PATH/data/$caseName/output/"*

    #--------------------------------------------------------------------------

    echo "*** EVOLVING ***"
    echo "Output is being written to $ODT_PATH/$caseName/runtime/runtime_* and $ODT_PATH/$caseName/data"
    ./odt.x $caseName $realizationNum          # realizationNum is the shift (realization # here)

    #--------------------------------------------------------------------------

    # Task 7: copy data to 'output' directory for RL framework
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/statistics/stat_odt_end.dat $ODT_PATH/data/$caseName/output/statistics.dat
    cp $ODT_PATH/data/$caseName/data/data_$formattedRealizationNum/state/state_odt_end.dat $ODT_PATH/data/$caseName/output/state.dat

    # Task 8: remove input from previous ODT simulation
    rm "$ODT_PATH/data/$caseName/input/restart"*

}

###############################################################################

evolveCase

###############################################################################
echo
echo "the end simulation time is"
date
###############################################################################

exit 0