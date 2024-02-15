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
tEndIncrement=1.0

###############################################################################

evolveCase () {

    # Task 1: Set Lrestart to true in input.yaml
    input_yaml_path="../data/$caseName/input/input.yaml"
    sed -i 's/Lrestart:.*/Lrestart:       true/' "$input_yaml_path"

    # Task 2: Find and copy the file with the highest number in the data folder
    highest_file=$(ls -1v ../data/$caseName/data/data_00000/dmp_*.dat | tail -n 1)
    highest_file_stat=$(ls -1v ../data/$caseName/data/data_00000/statistics/dmp_*_stat.dat | tail -n 1)
    cp "$highest_file" ../data/$caseName/input/restart.dat
    cp "$highest_file_stat" ../data/$caseName/input/restartStat.dat

    # Task 3: Extract and compare the time value of last data snapshot with tEnd input parameter
    time_value=$(head -n 1 ../data/$caseName/input/restart.dat | grep -oP '# time = \K[0-9.]+')
    yaml_tEnd=$(awk '/tEnd:/ {print $2}' "$input_yaml_path")
    # Check if tEnd is a valid number
    if ! [[ "$yaml_tEnd" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: Invalid tEnd format in input.yaml"
        exit 1
    fi
    # Check if time_value and tEnd parameter are equal
    compare_result=$(echo "$time_value == $yaml_tEnd" | bc)
    if [ "$compare_result" -ne 1 ]; then
        echo "Error: Time value in restart file does not match params.tEnd in input.yaml"
        exit 1
    fi
    # Increment tEnd by tEndIncrement
    new_tEnd=$(echo "$yaml_tEnd + $tEndIncrement" | bc)
    # Update tEnd value in input.yaml
    sed -i "s/tEnd:           $yaml_tEnd/tEnd:           $new_tEnd/" "$input_yaml_path"
    
    # Task 4: update dTimeEnd as tEnd, to produce output data for additional simulation time
    dTimeEnd=$(awk '/dTimeEnd:/ {print $2}' "$input_yaml_path")
    sed -i "s/dTimeEnd:       $dTimeEnd/dTimeEnd:       $new_tEnd/" "$input_yaml_path"

    #--------------------------------------------------------------------------

    echo "*** EVOLVING ***"
    echo "Output is being written to ../$caseName/runtime/runtime_* and ../$caseName/data"
    ./odt.x $caseName 0          # 0 is the shift (realization # here)

}

###############################################################################

evolveCase "$caseName"

###############################################################################
echo
echo "the end simulation time is"
date
###############################################################################

exit 0