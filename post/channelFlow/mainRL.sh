#!/bin/bash

# Run as:
# ./mainRL.sh channel180_2_5_2024_longSimulation 0 channel180_2_5_2024 0 30 5 180 50

# Check if the correct number of arguments is provided
echo "$#"
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <1_case_name_nonRL> <2_realization_number_nonRL> <3_case_name_RL> <4_realization_number_min_RL> <5_realization_number_max_RL> <6_realization_number_step_RL> <7_reynolds_number> <8_time_end_averaging>"
    exit 1
fi

# For each realization, if needed:
# Generate dmp file for odt_end.dat file (idem. for statistics & state) if last dmp file time < end file time
python3 load_end_file_to_dmp_file.py "$1" "$2"
for ((rlzNum = $4; rlzNum <= $5; rlzNum+=$6)); do
    echo -e "\n\n\n----------------------------------------------------------------"
    echo -e "------------------- load_end_file_to _dmp_file.py ---------------------"
    echo -e "----------------------------------------------------------------\n\n\n"
    python3 load_end_file_to_dmp_file.py "$3" "$rlzNum"
done

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ RL_realization_results.py -------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 RL_realization_results.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
