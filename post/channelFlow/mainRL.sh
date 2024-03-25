#!/bin/bash

# INSTRUCTIONS
# 
# Run as (e.g.):
# ./mainRL.sh 180 channel180_2_5_2024_longSimulation 0 channel180_2_5_2024 0 0 1 101 111 1000
# 
# Arguments:
# - 1_Re_tau 
# - 2_case_name_nonRL
# - 3_realization_number_nonRL
# - 4_case_name_RL
# - 5_realization_number_min_RL
# - 6_realization_number_max_RL
# - 7_realization_number_step_RL
# - 8_time_begin_averaging_non_converged
# - 9_time_end_averaging_non_converged
# - 10_time_end_averaging_converged

# Check if the correct number of arguments is provided
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <1_Re_tau> <2_case_name_nonRL> <3_realization_number_nonRL> <4_case_name_RL> <5_realization_number_min_RL> <6_realization_number_max_RL> <7_realization_number_step_RL> <8_time_begin_averaging_non_converged> <9_time_end_averaging_non_converged> <10_time_end_averaging_converged>"
    exit 1
else
    echo -e "\n\n\n****************************************************************"
    echo -e "************************** mainRL.sh ***************************"
    echo -e "****************************************************************"
    echo "- Reynolds number: $1"
    echo "- Case name non-RL: $2"
    echo "- Realization number non-RL: $3"
    echo "- Case name RL: $4"
    echo "- Realization number min RL: $5"
    echo "- Realization number max RL: $6"
    echo "- Realization number step RL: $7"
    echo "- Time begin averaging: $8"
    echo "- Time end averaging non-converged: $9"
    echo "- Time end averaging converged: ${10}"
fi

### # For each realization, if needed:
### # Generate dmp file for odt_end.dat file (idem. for statistics & state) if last dmp file time < end file time
### echo -e "\n\n\n----------------------------------------------------------------"
### echo -e "----------------- load_end_file_to_dmp_file.py -----------------"
### echo -e "----------------------------------------------------------------\n\n\n"
### python3 load_end_file_to_dmp_file.py "$2" "$3"
### for ((rlzNum = $5; rlzNum <= $6; rlzNum+=$7)); do
###     python3 load_end_file_to_dmp_file.py "$4" "$rlzNum"
### done

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ RL_realization_results.py -------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 RL_realization_results.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ RL_run_actions_rewards.py -------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 RL_run_actions_rewards.py "$4" "$5" "$6" "$7" "$8" "$9"

