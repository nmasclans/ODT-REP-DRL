#!/bin/bash

# INSTRUCTIONS
# 
# Run as (e.g.):
# ./mainRL.sh 180 channel180_2_5_2024_longSimulation 0 channel180_2_5_2024 0 8 1 111 1000 1710481991
# 
# Arguments:
# - 1_Re_tau 
# - 2_case_name_nonRL
# - 3_realization_number_nonRL
# - 4_case_name_RL
# - 5_realization_number_min_RL
# - 6_realization_number_max_RL
# - 7_realization_number_step_RL
# - 8_time_end_averaging_non_converged
# - 9_time_end_averaging_converged
# - 10_RL_run_id

# Check if the correct number of arguments is provided
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <1_Re_tau> <2_case_name_nonRL> <3_realization_number_nonRL> <4_case_name_RL> <5_realization_number_min_RL> <6_realization_number_max_RL> <7_realization_number_step_RL> <8_time_end_averaging_non_converged> <9_time_end_averaging_converged> <10_RL_run_id>"
    exit 1
fi

# For each realization, if needed:
# Generate dmp file for odt_end.dat file (idem. for statistics & state) if last dmp file time < end file time
echo -e "\n\n\n----------------------------------------------------------------"
echo -e "----------------- load_end_file_to_dmp_file.py -----------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 load_end_file_to_dmp_file.py "$2" "$3"
for ((rlzNum = $5; rlzNum <= $6; rlzNum+=$7)); do
    python3 load_end_file_to_dmp_file.py "$4" "$rlzNum"
done

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ RL_realization_results.py -------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 RL_realization_results.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"

# echo -e "\n\n\n----------------------------------------------------------------"
# echo -e "------------------ RL_run_actions_rewards.py -------------------"
# echo -e "----------------------------------------------------------------\n\n\n"
# python3 RL_run_actions_rewards.py "$4" "$10"

